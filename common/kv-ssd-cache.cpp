// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// SSD-Backed KV Cache with Hot/Warm/Cold Tiering
// Per-checkpoint file storage with ring buffer eviction.

#include "kv-ssd-cache.h"
#include "kv-ssd-posix.h"

#include "log.h"

#include <cstring>
#include <cinttypes>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <fcntl.h>  // posix_fadvise
#endif

// macOS has posix_fadvise via fcntl.h but defines it differently
#ifdef __APPLE__
#include <fcntl.h>
#endif

#include "host-ram.h"

// Magic numbers
static const uint32_t KV_SSD_MAGIC_INDEX = 0x4B564944; // "KVID"
static const uint32_t KV_SSD_MAGIC_REC   = 0x4B565243; // "KVRC"
// v3 = per-file format + dft/spec blobs
// v4 = extended metadata (model_identity, model_hash, quantization,
//      payload_size, header_checksum) in index header and per-checkpoint
//      header. Bumping the version forces a clean miss for v3 caches
//      rather than attempting to deserialize incompatible state.
//      KV_SSD_CACHE_FORMAT_VERSION is the inner version of the cache
//      contents payload (currently 1, bumped only when the payload
//      layout itself changes - rare). The header version is
//      KV_SSD_VERSION.
static const uint32_t KV_SSD_VERSION              = 4;
static const uint32_t KV_SSD_CACHE_FORMAT_VERSION = 1;

// =============================================================================
// Internal helpers
// =============================================================================

static uint64_t now_ms() {
    auto tp = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()).count();
}

// FNV-1a hash for token sequences
static uint64_t kv_ssd_hash_token_update(uint64_t h, uint32_t v) {
    h ^= (uint64_t)(v & 0xFF);
    h *= 1099511628211ULL;
    h ^= (uint64_t)((v >> 8) & 0xFF);
    h *= 1099511628211ULL;
    h ^= (uint64_t)((v >> 16) & 0xFF);
    h *= 1099511628211ULL;
    h ^= (uint64_t)((v >> 24) & 0xFF);
    h *= 1099511628211ULL;
    return h;
}

uint64_t kv_ssd_hash_tokens(const uint32_t* tokens, size_t count) {
    if (!tokens || count == 0) return 0;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < count; i++) {
        h = kv_ssd_hash_token_update(h, tokens[i]);
    }
    return h;
}

uint64_t kv_ssd_hash_conversation(
    const uint32_t* tokens,
    size_t tokens_size,
    size_t first_user_end) {
    if (!tokens || tokens_size == 0) return 0;
    const size_t stable_count = first_user_end > 0 && first_user_end <= tokens_size
        ? first_user_end
        : std::min(tokens_size, (size_t)1024);
    return kv_ssd_hash_tokens(tokens, stable_count);
}


// Write exactly `count` bytes to fd at offset.
// Chunks at 64 MiB because Windows _write/_read return int (32-bit)
// and cannot transfer >2 GiB in a single call. Also handles platforms
// where ssize_t is 32-bit (MinGW) and large checkpoints (>=2 GiB).
// Offsets are int64_t, not off_t: MSVC's off_t is 32-bit and wraps
// negative once a checkpoint crosses 2 GiB.
static bool pwrite_all(int fd, const void* buf, size_t count, int64_t offset) {
    static const size_t chunk_max = 64 * 1024 * 1024; // 64 MiB
    const char* ptr = (const char*)buf;
    size_t remaining = count;
    int64_t off = offset;
    while (remaining > 0) {
        size_t chunk = remaining;
        if (chunk > chunk_max) {
            chunk = chunk_max;
        }
        ssize_t n = pwrite(fd, ptr, chunk, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) {
            errno = ENOSPC;
            return false;
        }
        ptr += n;
        off += n;
        remaining -= (size_t)n;
    }
    return true;
}

// Read exactly `count` bytes from fd at offset.
// Chunks at 64 MiB for the same reason as pwrite_all.
static bool pread_all(int fd, void* buf, size_t count, int64_t offset) {
    static const size_t chunk_max = 64 * 1024 * 1024; // 64 MiB
    char* ptr = (char*)buf;
    size_t remaining = count;
    int64_t off = offset;
    while (remaining > 0) {
        size_t chunk = remaining;
        if (chunk > chunk_max) {
            chunk = chunk_max;
        }
        ssize_t n = pread(fd, ptr, chunk, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) { errno = EIO; return false; } // unexpected EOF
        ptr += n;
        off += n;
        remaining -= (size_t)n;
    }
    return true;
}

// Get checkpoint file path: {model_dir}/ckpt-{id}.bin
static std::string ckpt_path(const kv_ssd_cache* c, uint64_t id) {
    return c->model_dir + "/ckpt-" + std::to_string(id) + ".bin";
}

// Get index file path: {model_dir}/index.bin
static std::string index_path(const kv_ssd_cache* c) {
    return c->model_dir + "/index.bin";
}

// FNV-1a 64-bit hash of a contiguous byte range. Used for header
// checksums (corruption detection, not authentication). A non-cryptographic
// hash is the right primitive here: this is verifying that bytes on
// disk still match what was written, not authenticating the writer.
static uint64_t fnv1a_bytes(const void * data, size_t count) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < count; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Validate a header checksum on read-back. Returns true if the checksum
// is absent (zero, meaning the writer didn't set it) or if it matches
// the recomputed value. Both kv_ssd_index_header and kv_ssd_record
// carry a uint64_t header_checksum field, so a template covers both.
template<typename T>
static bool validate_checksum(const T & hdr, const char * label) {
    if (hdr.header_checksum == 0) return true;
    T tmp = hdr;
    tmp.header_checksum = 0;
    const uint64_t expected = fnv1a_bytes(&tmp, sizeof(tmp));
    if (expected != hdr.header_checksum) {
        LOG_WRN("SSD cache: %s header checksum mismatch "
                "(stored=0x%016llx computed=0x%016llx) - corrupted\n",
                label,
                (unsigned long long) hdr.header_checksum,
                (unsigned long long) expected);
        return false;
    }
    return true;
}

static bool record_payload_sizes(
        const kv_ssd_record & rec,
        size_t & tgt_size,
        size_t & dft_size,
        size_t & spec_size,
        size_t & total_size) {
    if (rec.data_size > SIZE_MAX || rec.dft_data_size > SIZE_MAX || rec.spec_data_size > SIZE_MAX) {
        return false;
    }
    tgt_size = (size_t) rec.data_size;
    dft_size = (size_t) rec.dft_data_size;
    spec_size = (size_t) rec.spec_data_size;
    if (dft_size > SIZE_MAX - tgt_size || spec_size > SIZE_MAX - tgt_size - dft_size) {
        return false;
    }
    total_size = tgt_size + dft_size + spec_size;
    return (rec.payload_size == 0 || rec.payload_size == total_size) &&
        total_size <= (size_t) (INT64_MAX - (int64_t) sizeof(kv_ssd_record));
}

static bool record_file_size_matches(int fd, size_t payload_size) {
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) return false;
    const uint64_t expected = (uint64_t)sizeof(kv_ssd_record) + (uint64_t)payload_size;
    return (uint64_t)st.st_size == expected;
}

// A single (pointer, size) segment for scatter-write I/O.
struct write_segment {
    const void * data;
    size_t size;
};

// Write `data` (size bytes) atomically to `path`. The implementation:
//   1. Open {path}.tmp with O_WRONLY | O_CREAT | O_EXCL (no clobber).
//   2. Write the data.
//   3. fsync the data file so its contents are durable.
//   4. Close the fd.
//   5. std::filesystem::rename {path}.tmp -> {path} (atomic on the
//      same filesystem on POSIX; on Windows, MoveFileEx semantics).
//   6. If a parent-directory fsync is requested and supported, fsync
//      the parent so the rename is durable across a crash.
//
// If the process dies between any of these steps, the destination
// file is left untouched (the prior contents remain valid) and the
// orphan {path}.tmp can be deleted on next startup.
//
// Returns true on success, false on any I/O failure (in which case
// the temp file is removed).
static bool atomic_write_segments(const std::string & path,
                                  const std::vector<write_segment> & segments,
                                  bool do_fsync = true) {
    namespace fs = std::filesystem;
    const std::string tmp = path + ".tmp";
    // Best-effort cleanup of a stale tmp from a prior crashed write.
    // Errors here are non-fatal.
    std::error_code ec;
    fs::remove(tmp, ec);

    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        int se = errno;
        LOG_WRN("SSD cache: atomic_write open(%s) failed: %s (errno=%d)\n",
                tmp.c_str(), strerror(se), se);
        return false;
    }
    // Write each segment sequentially at the running offset. This
    // avoids buffering the entire payload (header + data + specs)
    // into a single buffer when the caller already has the data
    // scattered across separate allocations.
    off_t offset = 0;
    for (const auto & seg : segments) {
        if (seg.size > 0 && !pwrite_all(fd, seg.data, seg.size, offset)) {
            int se = errno;
            LOG_WRN("SSD cache: atomic_write pwrite(%s) failed: %s (errno=%d)\n",
                    tmp.c_str(), strerror(se), se);
            ::close(fd);
            ::unlink(tmp.c_str());
            return false;
        }
        offset += (off_t) seg.size;
    }
    if (do_fsync) {
        ::fsync(fd);
    }
    ::close(fd);
    // rename(2) on POSIX is atomic on the same filesystem.
    std::error_code ren_ec;
    fs::rename(tmp, path, ren_ec);
    if (ren_ec) {
        LOG_WRN("SSD cache: atomic_write rename(%s -> %s) failed: %s\n",
                tmp.c_str(), path.c_str(), ren_ec.message().c_str());
        ::unlink(tmp.c_str());
        return false;
    }
    // Best-effort parent-dir fsync. Not all filesystems support this;
    // ignore ENOTSUP / EINVAL.
    if (do_fsync) {
        const fs::path parent = fs::path(path).parent_path();
        if (!parent.empty()) {
            int pfd = ::open(parent.string().c_str(), O_RDONLY);
            if (pfd >= 0) {
                if (::fsync(pfd) != 0) {
                    // ENOTSUP on some filesystems (e.g. tmpfs) - fine.
                    if (errno != ENOTSUP && errno != EINVAL) {
                        LOG_DBG("SSD cache: parent fsync(%s) errno=%d (%s)\n",
                                parent.string().c_str(), errno, strerror(errno));
                    }
                }
                ::close(pfd);
            }
        }
    }
    return true;
}

// Convenience wrapper for the common single-segment case.
static bool atomic_write_file(const std::string & path,
                              const void * data, size_t size,
                              bool do_fsync = true) {
    if (size > 0) {
        return atomic_write_segments(path, { {data, size} }, do_fsync);
    }
    return atomic_write_segments(path, {}, do_fsync);
}

// Hint to the kernel that a checkpoint file will be needed soon.
// On Linux, uses posix_fadvise(POSIX_FADV_WILLNEED) to trigger
// async page cache prefetch. On other platforms, uses readahead().
// This overlaps SSD I/O with CPU work (token matching, etc.).
// Thread-safe: takes the cache mutex only for the index lookup.
static void ckpt_readahead(kv_ssd_cache* c, uint64_t id) {
    std::string path;
    off_t total_size = 0;
    bool need_prefetch = false;

    {
        std::lock_guard<std::mutex> lock(c->mutex);
        auto it = c->index.find(id);
        if (it == c->index.end()) return;

        // Already in RAM - no need to prefetch
        if (it->second.tier == KV_TIER_HOT || it->second.tier == KV_TIER_WARM) return;

        path = ckpt_path(c, id);
        total_size = (off_t)(sizeof(kv_ssd_record) + it->second.data_size
                             + it->second.dft_data_size + it->second.spec_data_size);
        need_prefetch = true;
    }
    // Mutex released - path and total_size are local copies

    if (!need_prefetch) return;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;

#ifdef __linux__
    posix_fadvise(fd, 0, total_size, POSIX_FADV_WILLNEED);
/// macOS does not expose readahead(); use fcntl(F_RDADVISE) instead.
#elif defined(__APPLE__)
    struct radvisory ra = { 0, (int)total_size };
    fcntl(fd, F_RDADVISE, &ra);
#endif

    close(fd);
}

// =============================================================================
// Index persistence
// =============================================================================

// Write the index file (next_id + compat_hash).
static bool write_index_file(kv_ssd_cache* c) {
    kv_ssd_index_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = KV_SSD_MAGIC_INDEX;
    hdr.version = KV_SSD_VERSION;
    hdr.next_id = c->next_id;
    hdr.compat_hash = c->compat_hash;
    hdr.cache_format_version = KV_SSD_CACHE_FORMAT_VERSION;
    hdr.quantization        = c->config.quantization;
    hdr.model_identity      = c->config.model_identity;
    hdr.model_hash          = c->config.model_hash;
    hdr.payload_size        = 0;  // index is fixed-size for now
    for (size_t i = 0; i < c->prefix_checkpoint_ids.size() && i < 3; ++i) {
        hdr.prefix_checkpoint_ids[i] = c->prefix_checkpoint_ids[i];
    }
    // FNV-1a over the header bytes with the checksum field zeroed. The
    // checksum is at a fixed offset; the prior zeroed state means
    // recomputing after the actual write gives a stable value.
    hdr.header_checksum = 0;
    hdr.header_checksum = fnv1a_bytes(&hdr, sizeof(hdr));

    // Atomic write: tmp + fsync + rename, so a SIGKILL mid-write
    // leaves the prior index file intact. See atomic_write_file().
    const std::string path = index_path(c);
    return atomic_write_file(path, &hdr, sizeof(hdr),
                             /*do_fsync=*/!c->config.no_fsync);
}

// Read the index file. Returns false if not found or invalid.
static bool read_index_file(kv_ssd_cache* c) {
    std::string path = index_path(c);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    kv_ssd_index_header hdr;
    bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
    close(fd);
    if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX || hdr.version != KV_SSD_VERSION) return false;

    // Validate the v4 metadata if the writer set it. A zero
    // model_identity in either the header or the config means the
    // field is unset and we skip the check (legacy behavior).
    if (c->config.model_identity != 0 && hdr.model_identity != 0 &&
        hdr.model_identity != c->config.model_identity) {
        LOG_WRN("SSD cache: index model_identity mismatch "
                "(header=0x%016llx expected=0x%016llx) - clean miss\n",
                (unsigned long long) hdr.model_identity,
                (unsigned long long) c->config.model_identity);
        return false;
    }

    // Validate header checksum. If the writer set it, we require it
    // to match; otherwise (zero) we skip.
    if (!validate_checksum(hdr, "index")) return false;

    c->next_id = hdr.next_id;
    c->prefix_checkpoint_ids.clear();
    for (uint64_t id : hdr.prefix_checkpoint_ids) {
        if (id != 0 && c->config.prefix_checkpoints > 0 &&
                std::find(c->prefix_checkpoint_ids.begin(), c->prefix_checkpoint_ids.end(), id) ==
                c->prefix_checkpoint_ids.end()) {
            c->prefix_checkpoint_ids.push_back(id);
        }
    }
    while (c->prefix_checkpoint_ids.size() > (size_t)std::max(0, c->config.prefix_checkpoints)) {
        c->prefix_checkpoint_ids.erase(c->prefix_checkpoint_ids.begin());
    }
    if (c->compat_hash == 0) c->compat_hash = hdr.compat_hash;
    // Cache the metadata for downstream consumers (per-checkpoint
    // files inherit it; mismatched files will be rejected).
    if (c->model_identity == 0) c->model_identity = hdr.model_identity;
    if (c->quantization    == 0) c->quantization    = hdr.quantization;
    if (c->model_hash      == 0) c->model_hash      = hdr.model_hash;
    return true;
}

// Load a single checkpoint file into the in-memory index.
static bool load_checkpoint_file(kv_ssd_cache* c, const std::string& filepath) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;

    kv_ssd_record rec;
    bool ok = pread_all(fd, &rec, sizeof(rec), 0);
    close(fd);
    if (!ok || rec.magic != KV_SSD_MAGIC_REC) return false;
    // Reject per-checkpoint v3/v4 mismatches. v3 records are not
    // compatible with the v4 reader because the v4 fields overlap
    // the v3 reserved slot and would be read as garbage. Clean
    // miss is the right behavior.
    if (rec.version != KV_SSD_VERSION || rec.id == 0 || rec.id == UINT64_MAX ||
        rec.token_count > KV_SSD_TOKEN_PREFIX_MAX ||
        rec.data_size > SIZE_MAX || rec.dft_data_size > SIZE_MAX || rec.spec_data_size > SIZE_MAX) {
        return false;
    }
    // Header checksum validation. Skip if the writer didn't set one
    // (zero), require a match if it did.
    if (!validate_checksum(rec, filepath.c_str())) return false;
    // Per-checkpoint model identity check. If either the cache config
    // or the checkpoint header carries a non-zero identity and they
    // differ, reject the checkpoint. A legacy v4 record may have
    // identity == 0 (writer didn't set it) - we accept those for
    // backwards compatibility within v4, but the index header must
    // still match (enforced in read_index_file).
    if (rec.model_identity != 0 &&
        c->model_identity != 0 &&
        rec.model_identity != c->model_identity) {
        LOG_DBG("SSD cache: checkpoint %s model_identity 0x%016llx "
                "does not match cache 0x%016llx - skipping\n",
                filepath.c_str(),
                (unsigned long long) rec.model_identity,
                (unsigned long long) c->model_identity);
        return false;
    }

    // Build index entry
    kv_ssd_checkpoint ckpt;
    ckpt.id = rec.id;
    ckpt.slot_id = rec.slot_id;
    ckpt.pos_min = rec.pos_min;
    ckpt.pos_max = rec.pos_max;
    ckpt.n_tokens = rec.n_tokens;
    ckpt.turn_created = rec.turn_created;
    ckpt.turn_id = rec.turn_created;
    ckpt.token_hash = rec.token_hash;
    ckpt.compat_hash = rec.compat_hash;
    ckpt.token_count = (size_t)rec.token_count;
    ckpt.tier = KV_TIER_COLD;
    ckpt.data_size = (size_t)rec.data_size;
    ckpt.dft_data_size  = (size_t)rec.dft_data_size;
    ckpt.spec_data_size = (size_t)rec.spec_data_size;
    ckpt.last_access = 0;
    ckpt.access_count = 0;
    ckpt.model_identity = rec.model_identity;
    ckpt.quantization   = rec.quantization;
    ckpt.model_hash     = rec.model_hash;

    if (rec.token_count > 0) {
        ckpt.token_prefix.assign(rec.token_prefix, rec.token_prefix + rec.token_count);
    }

    c->index[rec.id] = ckpt;
    auto latest = c->slot_latest.find(rec.slot_id);
    if (latest == c->slot_latest.end()) {
        c->slot_latest[rec.slot_id] = rec.id;
    } else {
        const auto & previous = c->index.at(latest->second);
        if (ckpt.turn_created > previous.turn_created ||
            (ckpt.turn_created == previous.turn_created && rec.id > previous.id)) {
            latest->second = rec.id;
        }
    }
    return true;
}

// Scan the model directory for ckpt-*.bin files and load their headers.
static size_t scan_checkpoint_files(kv_ssd_cache* c) {
    namespace fs = std::filesystem;
    fs::path dir(c->model_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) return 0;

    size_t loaded = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string fname = entry.path().filename().string();
        if (fname.size() > 4 && fname.compare(fname.size() - 4, 4, ".tmp") == 0) {
            std::error_code time_ec;
            const auto modified = fs::last_write_time(entry.path(), time_ec);
            if (time_ec || modified + std::chrono::minutes(10) >= fs::file_time_type::clock::now()) {
                continue;
            }
            std::error_code cleanup_ec;
            fs::remove(entry.path(), cleanup_ec);
            if (cleanup_ec) {
                LOG_WRN("SSD cache: failed to remove stale temporary file %s: %s\n",
                        entry.path().string().c_str(), cleanup_ec.message().c_str());
            }
            continue;
        }
        if (fname.size() < 9) continue;
        if (fname.compare(0, 5, "ckpt-") != 0) continue;
        if (fname.compare(fname.size() - 4, 4, ".bin") != 0) continue;

        std::string filepath = (dir / fname).string();
        if (load_checkpoint_file(c, filepath)) {
            loaded++;
        } else {
            LOG_WRN("SSD cache: warning: failed to load %s\n", filepath.c_str());
        }
    }
    return loaded;
}

// Delete a checkpoint file from disk.
static bool delete_checkpoint_file(kv_ssd_cache* c, uint64_t id) {
    std::string path = ckpt_path(c, id);
    if (unlink(path.c_str()) != 0 && errno != ENOENT) {
        int se = errno;
        LOG_WRN("SSD cache: failed to delete %s: %s (errno=%d)\n", path.c_str(), strerror(se), se);
        return false;
    }
    return true;
}

// Remove an index entry and any resident blob. Caller must hold c->mutex.
static bool forget_checkpoint_locked(kv_ssd_cache* c, uint64_t id) {
    auto ckpt_it = c->index.find(id);
    if (ckpt_it == c->index.end()) return false;

    auto hot_it = c->hot_cache.find(id);
    if (hot_it != c->hot_cache.end()) {
        c->hot_bytes -= hot_it->second.size();
        c->hot_cache.erase(hot_it);
    }
    auto warm_it = c->warm_cache.find(id);
    if (warm_it != c->warm_cache.end()) {
        c->warm_bytes -= warm_it->second.size();
        c->warm_cache.erase(warm_it);
    }
    c->index.erase(ckpt_it);
    auto & anchors = c->prefix_checkpoint_ids;
    anchors.erase(std::remove(anchors.begin(), anchors.end(), id), anchors.end());

    for (auto latest_it = c->slot_latest.begin(); latest_it != c->slot_latest.end();) {
        if (latest_it->second != id) {
            ++latest_it;
            continue;
        }
        const uint32_t slot_id = latest_it->first;
        uint64_t replacement_id = 0;
        uint32_t replacement_turn = 0;
        for (const auto& [candidate_id, candidate] : c->index) {
            if (candidate.slot_id != slot_id) continue;
            if (replacement_id == 0 || candidate.turn_created > replacement_turn ||
                (candidate.turn_created == replacement_turn && candidate_id > replacement_id)) {
                replacement_id = candidate_id;
                replacement_turn = candidate.turn_created;
            }
        }
        if (replacement_id != 0) {
            latest_it->second = replacement_id;
            ++latest_it;
        } else {
            latest_it = c->slot_latest.erase(latest_it);
        }
    }
    return true;
}

// =============================================================================
// Ring buffer eviction (internal, caller must hold mutex)
// =============================================================================

// Evict oldest cold checkpoints until count <= max_cold_checkpoints.
// Deletes files from disk and removes from in-memory index.
static void ring_buffer_evict(kv_ssd_cache* c) {
    int max_cold = c->config.max_cold_checkpoints;
    if (max_cold <= 0) return;

    // Count cold entries
    size_t cold_count = 0;
    for (const auto& [id, ckpt] : c->index) {
        if (ckpt.tier == KV_TIER_COLD) cold_count++;
    }

    if ((int)cold_count <= max_cold) return;

    // Sort cold entries by turn_created (oldest first)
    std::vector<std::pair<uint64_t, uint32_t>> cold_by_age; // (id, turn_created)
    for (const auto& [id, ckpt] : c->index) {
        if (ckpt.tier == KV_TIER_COLD) {
            cold_by_age.emplace_back(id, ckpt.turn_created);
        }
    }
    std::sort(cold_by_age.begin(), cold_by_age.end(),
        [c](const auto& a, const auto& b) {
            const auto & anchors = c->prefix_checkpoint_ids;
            const auto pa = std::find(anchors.begin(), anchors.end(), a.first);
            const auto pb = std::find(anchors.begin(), anchors.end(), b.first);
            if ((pa != anchors.end()) != (pb != anchors.end())) return pa == anchors.end();
            if (pa != anchors.end() && pb != anchors.end()) return pa < pb;
            return a.second != b.second ? a.second < b.second : a.first < b.first;
        });

    int to_evict = (int)cold_count - max_cold;
    int evicted = 0;
    for (int i = 0; i < to_evict && i < (int)cold_by_age.size(); i++) {
        uint64_t id = cold_by_age[i].first;
        auto ckpt_it = c->index.find(id);
        if (ckpt_it == c->index.end()) continue;

        // Delete file from disk
        if (!delete_checkpoint_file(c, id)) continue;
        forget_checkpoint_locked(c, id);
        evicted++;
    }

    if (evicted > 0) {
        write_index_file(c);
        LOG_INF("SSD cache: ring buffer evicted %d checkpoints (limit=%d, remaining=%zu)\n",
                evicted, max_cold, c->index.size());
    }
}

// =============================================================================
// Tier management (internal, caller must hold mutex)
// =============================================================================

// Demote LRU hot entries to warm until hot_bytes <= budget.
static void demote_hot_to_warm(kv_ssd_cache* c) {
   while (c->hot_bytes > c->config.hot_ram_bytes && !c->hot_cache.empty()) {
       uint64_t lru_id = 0;
       uint64_t lru_time = UINT64_MAX;

       for (const auto& [id, ckpt] : c->index) {
            if (ckpt.tier == KV_TIER_HOT && ckpt.last_access < lru_time) {
                lru_time = ckpt.last_access;
                lru_id = id;
            }
       }
        if (lru_id == 0) break;

        auto it = c->hot_cache.find(lru_id);
        if (it == c->hot_cache.end()) {
            c->index[lru_id].tier = KV_TIER_COLD;
            continue;
        }

        size_t sz = it->second.size();
        c->warm_cache[lru_id] = std::move(it->second);
        c->hot_cache.erase(it);
        c->hot_bytes -= sz;
        c->warm_bytes += sz;
        c->index[lru_id].tier = KV_TIER_WARM;

        LOG_INF("SSD cache: demoted checkpoint %lu hot->warm (hot=%zu MiB, warm=%zu MiB)\n",
                (unsigned long)lru_id, c->hot_bytes / 1024 / 1024, c->warm_bytes / 1024 / 1024);
    }
}

// Demote LRU warm entries to cold (data stays on disk).
static void demote_warm_to_cold(kv_ssd_cache* c) {
    while (c->warm_bytes > c->config.warm_ram_bytes && !c->warm_cache.empty()) {
        uint64_t lru_id = 0;
        uint64_t lru_time = UINT64_MAX;

        for (const auto& [id, ckpt] : c->index) {
            if (ckpt.tier == KV_TIER_WARM && ckpt.last_access < lru_time) {
                lru_time = ckpt.last_access;
                lru_id = id;
            }
        }
        if (lru_id == 0) break;

        auto it = c->warm_cache.find(lru_id);
        if (it == c->warm_cache.end()) {
            c->index[lru_id].tier = KV_TIER_COLD;
            continue;
        }

        size_t sz = it->second.size();
        c->warm_cache.erase(it);
        c->warm_bytes -= sz;
        c->index[lru_id].tier = KV_TIER_COLD;

        LOG_INF("SSD cache: demoted checkpoint %lu warm->cold (warm=%zu MiB)\n",
                (unsigned long)lru_id, c->warm_bytes / 1024 / 1024);
    }
}

// Make room in hot tier for `needed` bytes.
static void make_room_hot(kv_ssd_cache* c, size_t needed) {
    while (c->hot_bytes + needed > c->config.hot_ram_bytes && !c->hot_cache.empty()) {
        demote_hot_to_warm(c);
        if (c->hot_bytes + needed <= c->config.hot_ram_bytes) break;
        demote_warm_to_cold(c);
        demote_hot_to_warm(c);
        if (c->hot_bytes + needed > c->config.hot_ram_bytes) break;
    }
}

// Promote a checkpoint to hot tier (load from SSD file if needed).
static bool promote_to_hot(kv_ssd_cache* c, uint64_t id, std::vector<uint8_t>* uncached_blob = nullptr) {
    auto it = c->index.find(id);
    if (it == c->index.end()) return false;

    auto& ckpt = it->second;

    // Already hot
    if (ckpt.tier == KV_TIER_HOT) {
        ckpt.last_access = now_ms();
        ckpt.access_count++;
        return true;
    }

    // If warm, move to hot
    if (ckpt.tier == KV_TIER_WARM) {
        auto wit = c->warm_cache.find(id);
        if (wit != c->warm_cache.end()) {
            // Move data out BEFORE make_room_hot. make_room_hot calls
            // demote_warm_to_cold with prefer_conv_hash, which can evict
            // THIS entry from warm_cache — invalidating wit. Moving
            // data first prevents use-after-erase segfault.
            auto data = std::move(wit->second);
            size_t sz = data.size();
            c->warm_cache.erase(wit);
            c->warm_bytes -= sz;

            if (sz > c->config.hot_ram_bytes) {
                ckpt.tier = KV_TIER_COLD;
                ckpt.last_access = now_ms();
                ckpt.access_count++;
                if (uncached_blob) *uncached_blob = std::move(data);
                return uncached_blob != nullptr;
            }

            make_room_hot(c, sz);

            c->hot_cache[id] = std::move(data);
            c->hot_bytes += sz;
            ckpt.tier = KV_TIER_HOT;
            ckpt.last_access = now_ms();
            ckpt.access_count++;
            return true;
        }
    }

    // Cold - read from SSD file
    std::string filepath = ckpt_path(c, id);

    // Hint kernel to prefetch the file before we read it.
    // This is a no-op if kv_ssd_prefetch was already called for this checkpoint.
    // Note: ckpt_readahead takes the mutex, but we already hold it here.
    // Instead, we do the fadvise inline since we're already locked.
#ifdef __linux__
    {
        int ra_fd = open(filepath.c_str(), O_RDONLY);
        if (ra_fd >= 0) {
            size_t total = sizeof(kv_ssd_record) + ckpt.data_size
                           + ckpt.dft_data_size + ckpt.spec_data_size;
            posix_fadvise(ra_fd, 0, (off_t)total, POSIX_FADV_WILLNEED);
            close(ra_fd);
        }
    }
#endif

    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_WRN("SSD cache: checkpoint file %s not found\n", filepath.c_str());
        // File was deleted (ring buffer evicted) but index still had entry - clean up
        forget_checkpoint_locked(c, id);
        return false;
    }

    // Read record header
    kv_ssd_record rec;
    if (!pread_all(fd, &rec, sizeof(rec), 0) || rec.magic != KV_SSD_MAGIC_REC || rec.id != id) {
        close(fd);
        return false;
    }
    // v4: also reject the read on version mismatch or checksum
    // failure. Same rules as the index load - corrupt or
    // version-incompatible checkpoints are treated as a clean miss.
    if (rec.version != KV_SSD_VERSION) {
        LOG_WRN("SSD cache: checkpoint id=%lu version=%u (expected %u) - skipping\n",
                (unsigned long) id, rec.version, KV_SSD_VERSION);
        close(fd);
        return false;
    }
    if (!validate_checksum(rec, "checkpoint")) {
        close(fd);
        return false;
    }
    if (rec.model_identity != 0 && c->model_identity != 0 &&
        rec.model_identity != c->model_identity) {
        LOG_WRN("SSD cache: checkpoint id=%lu model_identity 0x%016llx "
                "does not match active model 0x%016llx - skipping\n",
                (unsigned long) id,
                (unsigned long long) rec.model_identity,
                (unsigned long long) c->model_identity);
        close(fd);
        return false;
    }

    size_t tgt_size = 0;
    size_t dft_size = 0;
    size_t spec_size = 0;
    size_t total_blob = 0;
    if (!record_payload_sizes(rec, tgt_size, dft_size, spec_size, total_blob)) {
        LOG_WRN("SSD cache: checkpoint id=%lu has invalid payload sizes\n", (unsigned long) id);
        close(fd);
        return false;
    }
    if (!record_file_size_matches(fd, total_blob)) {
        LOG_WRN("SSD cache: checkpoint id=%lu payload length does not match file size\n", (unsigned long) id);
        close(fd);
        return false;
    }
    if (total_blob <= c->config.hot_ram_bytes) {
        make_room_hot(c, total_blob);
    }

    // Read all blobs concatenated: [tgt_data][dft_data][spec_data]
    std::vector<uint8_t> data(total_blob);
    if (!pread_all(fd, data.data(), total_blob, (int64_t)sizeof(kv_ssd_record))) {
        close(fd);
        return false;
    }
    close(fd);

    // Update index split sizes in case they differed from what was recovered
    ckpt.dft_data_size  = dft_size;
    ckpt.spec_data_size = spec_size;

    ckpt.last_access = now_ms();
    ckpt.access_count++;
    c->stats_loads++;

    if (total_blob > c->config.hot_ram_bytes) {
        ckpt.tier = KV_TIER_COLD;
        if (uncached_blob) *uncached_blob = std::move(data);
        LOG_INF("SSD cache: loaded oversized checkpoint %lu without RAM pin (%zu MiB, hot limit=%zu MiB)\n",
                (unsigned long)id, total_blob / 1024 / 1024,
                c->config.hot_ram_bytes / 1024 / 1024);
        return uncached_blob != nullptr;
    }

    c->hot_cache[id] = std::move(data);
    c->hot_bytes += total_blob;
    ckpt.tier = KV_TIER_HOT;

    LOG_INF("SSD cache: promoted checkpoint %lu cold->hot (%zu MiB)\n",
            (unsigned long)id, total_blob / 1024 / 1024);
    return true;
}

// Read a checkpoint that exceeds the hot-tier budget directly into its target,
// draft, and speculative output vectors. This avoids a second combined payload
// allocation during native-context restores (roughly 10 GiB for this model).
// Caller must hold c->mutex.
static bool load_cold_direct(
        kv_ssd_cache * c,
        uint64_t id,
        std::vector<uint8_t> & out_data,
        std::vector<uint8_t> * out_dft_data,
        std::vector<uint8_t> * out_spec_data) {
    auto it = c->index.find(id);
    if (it == c->index.end()) return false;

    const std::string filepath = ckpt_path(c, id);
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        forget_checkpoint_locked(c, id);
        return false;
    }

    kv_ssd_record rec;
    size_t tgt_size = 0;
    size_t dft_size = 0;
    size_t spec_size = 0;
    size_t total_size = 0;
    if (!pread_all(fd, &rec, sizeof(rec), 0) ||
        rec.magic != KV_SSD_MAGIC_REC || rec.version != KV_SSD_VERSION || rec.id != id ||
        !validate_checksum(rec, "checkpoint") ||
        !record_payload_sizes(rec, tgt_size, dft_size, spec_size, total_size) ||
        !record_file_size_matches(fd, total_size) ||
        (rec.model_identity != 0 && c->model_identity != 0 && rec.model_identity != c->model_identity)) {
        close(fd);
        return false;
    }

    out_data.resize(tgt_size);
    if (out_dft_data) out_dft_data->resize(dft_size);
    if (out_spec_data) out_spec_data->resize(spec_size);

    int64_t offset = (int64_t) sizeof(kv_ssd_record);
    bool ok = pread_all(fd, out_data.data(), tgt_size, offset);
    offset += (int64_t) tgt_size;
    if (ok && out_dft_data && dft_size > 0) {
        ok = pread_all(fd, out_dft_data->data(), dft_size, offset);
    }
    offset += (int64_t) dft_size;
    if (ok && out_spec_data && spec_size > 0) {
        ok = pread_all(fd, out_spec_data->data(), spec_size, offset);
    }
    close(fd);
    if (!ok) return false;

    auto & checkpoint = it->second;
    checkpoint.data_size = tgt_size;
    checkpoint.dft_data_size = dft_size;
    checkpoint.spec_data_size = spec_size;
    checkpoint.tier = KV_TIER_COLD;
    checkpoint.last_access = now_ms();
    checkpoint.access_count++;
    c->stats_loads++;
    LOG_INF("SSD cache: streamed oversized checkpoint %lu directly from SSD (%zu MiB, hot limit=%zu MiB)\n",
            (unsigned long) id, total_size / 1024 / 1024,
            c->config.hot_ram_bytes / 1024 / 1024);
    return true;
}

// =============================================================================
// Public API
// =============================================================================

kv_ssd_cache* kv_ssd_init(const char* path, const kv_ssd_config* cfg, uint64_t conv_hash, const char* namespace_prefix) {
    namespace_prefix = namespace_prefix ? namespace_prefix : "";

    if (!path) return nullptr;

    kv_ssd_cache* c = new kv_ssd_cache();
    if (cfg) c->config = *cfg;
    c->conv_hash = conv_hash;
    // Carry v4 metadata into the cache state. read_index_file() may
    // refresh these from the on-disk index header.
    c->model_identity = c->config.model_identity;
    c->quantization   = c->config.quantization;
    c->model_hash     = c->config.model_hash;

    // Auto-size RAM budgets
    if (c->config.auto_size) {
        size_t avail = common::host_available_ram();
        // On UMA APUs the loaded model is GTT-mapped in system RAM and
        // MemAvailable counts those pages as evictable, which leads
        // auto-size to reserve RAM the model is actively using. Subtract
        // the model footprint when the caller provided it so the
        // hot+warm tiers don't compete with the GTT-mapped weights.
        // Cap the subtraction at (avail / 2) so a misconfigured
        // oversized model_size_bytes can't drive the budget to zero or
        // negative.
        if (c->config.model_size_bytes > 0) {
            size_t model_sub = c->config.model_size_bytes;
            size_t cap = avail / 2;
            if (model_sub > cap) model_sub = cap;
            if (model_sub < avail) avail -= model_sub;
        }
        size_t usable = (size_t)((double)avail * (1.0 - c->config.memory_reserve));
        c->config.hot_ram_bytes = (usable * 3) / 4;
        c->config.warm_ram_bytes = usable / 4;
        const size_t MIN_HOT  = 512ULL * 1024 * 1024;
        const size_t MIN_WARM = 256ULL * 1024 * 1024;
        bool boosted = false;
        if (c->config.hot_ram_bytes < MIN_HOT) {
            c->config.hot_ram_bytes = MIN_HOT;
            boosted = true;
        }
        if (c->config.warm_ram_bytes < MIN_WARM) {
            c->config.warm_ram_bytes = MIN_WARM;
            boosted = true;
        }
        LOG_INF("SSD cache: auto-sized hot=%zu MiB warm=%zu MiB (avail=%zu MiB)\n",
                c->config.hot_ram_bytes / 1024 / 1024,
                c->config.warm_ram_bytes / 1024 / 1024,
                avail / 1024 / 1024);
        if (boosted) {
            LOG_INF("SSD cache: floors applied (min hot=%zu MiB, min warm=%zu MiB)\n",
                    MIN_HOT / 1024 / 1024, MIN_WARM / 1024 / 1024);
        }
    }

    // Create base directory
    mkdir(path, 0755);

    // Create conversation-specific directory. namespace_prefix (e.g. "u/") is
    // appended to the hash hex so multiple identity schemes can coexist on
    // disk without colliding.
    char conv_hex[17];
    snprintf(conv_hex, sizeof(conv_hex), "%016" PRIx64, conv_hash);
    c->base_path = std::string(path);
    c->model_dir = std::string(path) + "/" + namespace_prefix + conv_hex;
    if (namespace_prefix[0] != '\0') {
        // Ensure the namespace subdir exists before creating the leaf.
        std::string ns_dir = std::string(path) + "/" + namespace_prefix;
        mkdir(ns_dir.c_str(), 0755);
    }
    mkdir(c->model_dir.c_str(), 0755);

    // Load index file for next_id
    if (!read_index_file(c)) {
        LOG_INF("SSD cache: no index file, starting fresh\n");
        c->index.clear();
        c->slot_latest.clear();
        c->next_id = 1;
    }

    // Scan checkpoint files to rebuild in-memory index
    size_t loaded = scan_checkpoint_files(c);
    auto & anchors = c->prefix_checkpoint_ids;
    anchors.erase(std::remove_if(anchors.begin(), anchors.end(),
        [c](uint64_t id) { return c->index.count(id) == 0; }), anchors.end());
    for (const auto& [id, checkpoint] : c->index) {
        (void)checkpoint;
        c->next_id = std::max(c->next_id, id + 1);
    }
    LOG_INF("SSD cache: loaded %zu checkpoints from %s (next_id=%lu)\n",
            loaded, c->model_dir.c_str(), (unsigned long)c->next_id);

    c->initialized = true;
    return c;
}

void kv_ssd_free(kv_ssd_cache* cache) {
    if (!cache) return;

    // Write final index file
    write_index_file(cache);

    LOG_INF("SSD cache: shutdown (stored=%lu hits=%lu misses=%lu evicts=%lu)\n",
            (unsigned long)cache->stats_stores,
            (unsigned long)cache->stats_hits,
            (unsigned long)cache->stats_misses,
            (unsigned long)cache->stats_evicts);

    delete cache;
}

void kv_ssd_set_compat_hash(kv_ssd_cache* cache, uint64_t compat_hash) {
    if (!cache) return;
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->compat_hash = compat_hash;
    LOG_INF("SSD cache: model compat_hash set to %016" PRIx64 "\n", compat_hash);
}

static uint64_t checkpoint_file_age_ms(const kv_ssd_cache* cache, uint64_t id) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto modified = fs::last_write_time(ckpt_path(cache, id), ec);
    if (ec) return UINT64_MAX;
    const auto now_file = fs::file_time_type::clock::now();
    if (modified >= now_file) return 0;
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        now_file - modified).count();
}

kv_ssd_store_plan kv_ssd_plan_store(
    kv_ssd_cache* cache,
    uint32_t slot_id,
    int32_t pos_min,
    int32_t pos_max,
    uint64_t n_tokens,
    uint32_t turn_id,
    const uint32_t* tokens,
    size_t tokens_size,
    uint64_t compat_hash,
    bool has_dft_state,
    bool has_spec_state,
    bool prefix_anchor) {
    kv_ssd_store_plan plan;
    if (!cache || !cache->initialized || !tokens || n_tokens == 0 || tokens_size < n_tokens) {
        return plan;
    }

    std::lock_guard<std::mutex> lock(cache->mutex);
    const size_t prefix_count = std::min((size_t)n_tokens, (size_t)KV_SSD_TOKEN_PREFIX_MAX);

    // Compute all candidate prefix hashes in one token pass. This keeps a
    // 262K-token plan O(tokens + checkpoints), rather than hashing the prompt
    // once per durable checkpoint while holding the cache lock.
    std::vector<size_t> hash_lengths = { (size_t)n_tokens };
    hash_lengths.reserve(cache->index.size() + 1);
    for (const auto & [id, checkpoint] : cache->index) {
        (void)id;
        if (checkpoint.n_tokens <= n_tokens) hash_lengths.push_back((size_t)checkpoint.n_tokens);
    }
    std::sort(hash_lengths.begin(), hash_lengths.end());
    hash_lengths.erase(std::unique(hash_lengths.begin(), hash_lengths.end()), hash_lengths.end());
    std::vector<uint64_t> prefix_hashes(hash_lengths.size());
    uint64_t running_hash = 14695981039346656037ULL;
    size_t next_length = 0;
    while (next_length < hash_lengths.size() && hash_lengths[next_length] == 0) {
        prefix_hashes[next_length++] = 0;
    }
    for (size_t i = 0; i < (size_t)n_tokens; ++i) {
        running_hash = kv_ssd_hash_token_update(running_hash, tokens[i]);
        while (next_length < hash_lengths.size() && hash_lengths[next_length] == i + 1) {
            prefix_hashes[next_length++] = running_hash;
        }
    }
    auto incoming_hash = [&](size_t count) {
        const auto it = std::lower_bound(hash_lengths.begin(), hash_lengths.end(), count);
        return prefix_hashes[(size_t)(it - hash_lengths.begin())];
    };
    const uint64_t token_hash = incoming_hash((size_t)n_tokens);

    uint64_t duplicate_id = 0;
    uint64_t base_id = 0;
    uint64_t base_n_tokens = 0;
    uint32_t base_turn = 0;

    for (auto& [id, checkpoint] : cache->index) {
        if (compat_hash != 0 && checkpoint.compat_hash != compat_hash) continue;
        if (cache->model_identity != 0 && checkpoint.model_identity != 0 &&
            checkpoint.model_identity != cache->model_identity) continue;
        if ((checkpoint.dft_data_size > 0) != has_dft_state) continue;
        if (checkpoint.n_tokens > n_tokens) continue;
        const size_t compare_count = std::min(prefix_count, (size_t)checkpoint.n_tokens);
        if (checkpoint.token_prefix.size() < compare_count) continue;
        if (compare_count > 0 &&
            std::memcmp(checkpoint.token_prefix.data(), tokens, compare_count * sizeof(uint32_t)) != 0) continue;
        if (checkpoint.token_hash != incoming_hash((size_t)checkpoint.n_tokens)) continue;

        if (checkpoint.n_tokens == n_tokens && checkpoint.token_hash == token_hash &&
            checkpoint.pos_min == pos_min && checkpoint.pos_max == pos_max &&
            ((checkpoint.spec_data_size > 0) == has_spec_state)) {
            if (duplicate_id == 0 || checkpoint.turn_created > cache->index[duplicate_id].turn_created ||
                (checkpoint.turn_created == cache->index[duplicate_id].turn_created && id > duplicate_id)) {
                duplicate_id = id;
            }
            continue;
        }

        if (checkpoint.n_tokens < n_tokens &&
            (base_id == 0 || checkpoint.n_tokens > base_n_tokens ||
             (checkpoint.n_tokens == base_n_tokens && checkpoint.turn_created > base_turn))) {
            base_id = id;
            base_n_tokens = checkpoint.n_tokens;
            base_turn = checkpoint.turn_created;
        }
    }

    if (duplicate_id != 0) {
        auto& checkpoint = cache->index[duplicate_id];
        checkpoint.turn_id = turn_id;
        checkpoint.last_access = now_ms();
        checkpoint.access_count++;
        cache->slot_latest[slot_id] = duplicate_id;
        plan.action = KV_SSD_STORE_REUSE;
        plan.checkpoint_id = duplicate_id;
        plan.base_n_tokens = n_tokens;
        return plan;
    }

    // A new shared preamble must have its own exact recurrent boundary;
    // ordinary growth/time suppression would discard it. Exact duplicates
    // above still avoid serialization and writes.
    if (prefix_anchor && cache->config.prefix_checkpoints > 0) return plan;

    const bool growth_enabled = cache->config.durable_min_growth_tokens > 0;
    const bool age_enabled = cache->config.durable_max_age_ms > 0;
    if (base_id == 0 || (!growth_enabled && !age_enabled)) {
        return plan;
    }

    plan.checkpoint_id = base_id;
    plan.base_n_tokens = base_n_tokens;
    plan.growth_tokens = n_tokens - base_n_tokens;
    plan.age_ms = checkpoint_file_age_ms(cache, base_id);
    const bool growth_due = growth_enabled &&
        plan.growth_tokens >= cache->config.durable_min_growth_tokens;
    const bool age_due = age_enabled && plan.age_ms >= cache->config.durable_max_age_ms;
    if (!growth_due && !age_due) {
        plan.action = KV_SSD_STORE_SKIP_CADENCE;
        auto & checkpoint = cache->index[base_id];
        checkpoint.turn_id = turn_id;
        checkpoint.last_access = now_ms();
        checkpoint.access_count++;
        cache->slot_latest[slot_id] = base_id;
    }
    return plan;
}

void kv_ssd_mark_prefix(kv_ssd_cache* cache, uint64_t checkpoint_id) {
    if (!cache || !cache->initialized || cache->config.prefix_checkpoints <= 0) return;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (cache->index.count(checkpoint_id) == 0) return;
    auto & anchors = cache->prefix_checkpoint_ids;
    if (!anchors.empty() && anchors.back() == checkpoint_id) return;
    anchors.erase(std::remove(anchors.begin(), anchors.end(), checkpoint_id), anchors.end());
    anchors.push_back(checkpoint_id);
    while (anchors.size() > (size_t)std::min(cache->config.prefix_checkpoints, 3)) {
        anchors.erase(anchors.begin());
    }
    write_index_file(cache);
}

uint64_t kv_ssd_store(kv_ssd_cache* cache,
                  uint32_t slot_id,
                  const uint8_t* data, size_t data_size,
                  int32_t pos_min, int32_t pos_max,
                  uint64_t n_tokens, uint32_t turn_id,
                  const uint32_t* tokens, size_t tokens_size,
                  uint64_t compat_hash,
                  const uint8_t* dft_data, size_t dft_data_size,
                  const uint8_t* spec_data, size_t spec_data_size)
{
    if (!cache || !cache->initialized || !data || data_size == 0) return 0;

    std::lock_guard<std::mutex> lock(cache->mutex);

    if (cache->next_id == 0 || cache->next_id == UINT64_MAX) {
        LOG_WRN("SSD cache: checkpoint ID space exhausted\n");
        return 0;
    }
    uint64_t id = cache->next_id++;

    const size_t checkpoint_tokens = std::min(tokens_size, (size_t)n_tokens);
    uint64_t token_hash = kv_ssd_hash_tokens(tokens, checkpoint_tokens);
    size_t token_count = tokens ? std::min(checkpoint_tokens, (size_t)KV_SSD_TOKEN_PREFIX_MAX) : 0;

    // Build record header
    kv_ssd_record rec;
    memset(&rec, 0, sizeof(rec));
    rec.magic = KV_SSD_MAGIC_REC;
    rec.version = KV_SSD_VERSION;
    rec.id = id;
    rec.slot_id = slot_id;
    rec.pos_min = pos_min;
    rec.pos_max = pos_max;
    rec.n_tokens = n_tokens;
    rec.turn_created = turn_id;
    rec.data_size = data_size;
    rec.token_hash = token_hash;
    rec.compat_hash = compat_hash;
    rec.token_count = (uint32_t)token_count;
    if (tokens && token_count > 0) {
        memcpy(rec.token_prefix, tokens, token_count * sizeof(uint32_t));
    }
    rec.dft_data_size  = (dft_data  && dft_data_size  > 0) ? dft_data_size  : 0;
    rec.spec_data_size = (spec_data && spec_data_size > 0) ? spec_data_size : 0;
    // v4 metadata. Inherits from the cache-level fields populated
    // from kv_ssd_config; if those are 0 the per-checkpoint fields
    // are also 0 and the read path skips the check.
    rec.cache_format_version = KV_SSD_CACHE_FORMAT_VERSION;
    rec.quantization        = cache->quantization;
    rec.model_identity      = cache->model_identity;
    rec.model_hash          = cache->model_hash;
    rec.payload_size        = data_size + rec.dft_data_size + rec.spec_data_size;
    rec.header_checksum     = 0;
    rec.header_checksum     = fnv1a_bytes(&rec, sizeof(rec));

    // Write checkpoint file atomically: build the full file in memory
    // then tmp + fsync + rename. Previously this used O_TRUNC + a
    // series of pwrite_all calls, which could leave a half-written
    // file if the process died between any of the writes. See
    // atomic_write_segments().
    std::string filepath = ckpt_path(cache, id);
    // Write segments directly to the tmp fd instead of buffering
    // the entire payload (header + data + dft + dft + spec) into a
    // std::vector<uint8_t>. For large checkpoints this avoids an
    // extra 10+ MiB allocation + memcpy per store.
    std::vector<write_segment> segments;
    segments.push_back({&rec, sizeof(rec)});
    if (data_size > 0)      segments.push_back({data,      data_size});
    if (rec.dft_data_size > 0) segments.push_back({dft_data, rec.dft_data_size});
    if (rec.spec_data_size > 0) segments.push_back({spec_data, rec.spec_data_size});
    if (!atomic_write_segments(filepath, segments,
                              /*do_fsync=*/!cache->config.no_fsync)) {
        cache->next_id--;
        return 0;
    }

    // Update index file with new next_id
    write_index_file(cache);

    // Build a resident combined blob only when the checkpoint fits the hot
    // budget. Oversized states remain cold; the active slot and kernel page
    // cache already hold the current data, so pinning another 5-10 GiB copy
    // would defeat the configured RAM limit.
    const size_t total_blob = data_size + rec.dft_data_size + rec.spec_data_size;
    const bool keep_hot = total_blob <= cache->config.hot_ram_bytes;
    if (keep_hot) {
        make_room_hot(cache, total_blob);
        std::vector<uint8_t> hot_blob;
        hot_blob.reserve(total_blob);
        hot_blob.assign(data, data + data_size);
        if (rec.dft_data_size > 0) hot_blob.insert(hot_blob.end(), dft_data, dft_data + rec.dft_data_size);
        if (rec.spec_data_size > 0) hot_blob.insert(hot_blob.end(), spec_data, spec_data + rec.spec_data_size);
        cache->hot_cache[id] = std::move(hot_blob);
        cache->hot_bytes += total_blob;
    }

    // Build index entry
    kv_ssd_checkpoint ckpt;
    ckpt.id = id;
    ckpt.slot_id = slot_id;
    ckpt.pos_min = pos_min;
    ckpt.pos_max = pos_max;
    ckpt.n_tokens = n_tokens;
    ckpt.turn_id = turn_id;
    ckpt.turn_created = turn_id;
    ckpt.token_hash = token_hash;
    ckpt.compat_hash = compat_hash;
    ckpt.token_count = token_count;
    ckpt.tier = keep_hot ? KV_TIER_HOT : KV_TIER_COLD;
    if (token_count > 0) {
        ckpt.token_prefix.assign(rec.token_prefix, rec.token_prefix + token_count);
    }
    ckpt.data_size      = data_size;
    ckpt.dft_data_size  = rec.dft_data_size;
    ckpt.spec_data_size = rec.spec_data_size;
    ckpt.model_identity = rec.model_identity;
    ckpt.quantization   = rec.quantization;
    ckpt.model_hash     = rec.model_hash;
    ckpt.last_access = now_ms();
    ckpt.access_count = 1;

    cache->index[id] = ckpt;
    cache->slot_latest[slot_id] = id;
    cache->stats_stores++;

    LOG_INF("SSD cache: stored checkpoint %lu slot=%u tokens=%lu tgt=%zu dft=%llu spec=%llu MiB "
            "tier=%s (hot=%zu MiB warm=%zu MiB total=%zu)\n",
            (unsigned long)id, slot_id, (unsigned long)n_tokens,
            data_size / 1024 / 1024,
            (unsigned long long)(rec.dft_data_size / 1024 / 1024),
            (unsigned long long)(rec.spec_data_size / 1024 / 1024), keep_hot ? "hot" : "cold",
            cache->hot_bytes / 1024 / 1024,
            cache->warm_bytes / 1024 / 1024,
            cache->index.size());

    return id;
}

bool kv_ssd_load(kv_ssd_cache* cache, uint64_t checkpoint_id,
                 std::vector<uint8_t>& out_data,
                 std::vector<uint8_t>* out_dft_data,
                 std::vector<uint8_t>* out_spec_data)
{
    if (!cache || !cache->initialized || checkpoint_id == 0) return false;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Reject checkpoints from incompatible model configs
    if (cache->compat_hash != 0) {
        auto ckpt_it = cache->index.find(checkpoint_id);
        if (ckpt_it != cache->index.end() && ckpt_it->second.compat_hash != cache->compat_hash) {
            LOG_WRN("SSD cache: rejecting checkpoint %lu - compat_hash mismatch "
                    "(stored=%016" PRIx64 " current=%016" PRIx64 ")\n",
                    (unsigned long)checkpoint_id,
                    ckpt_it->second.compat_hash,
                    cache->compat_hash);
            cache->stats_misses++;
            return false;
        }
    }

    // Helper: split combined blob [tgt][dft][spec] and populate output vectors
    auto split_blob = [&](const std::vector<uint8_t>& blob, const kv_ssd_checkpoint& ckpt) {
        out_data.assign(blob.begin(), blob.begin() + ckpt.data_size);
        if (out_dft_data) {
            if (ckpt.dft_data_size > 0) {
                out_dft_data->assign(blob.begin() + ckpt.data_size,
                                     blob.begin() + ckpt.data_size + ckpt.dft_data_size);
            } else {
                out_dft_data->clear();
            }
        }
        if (out_spec_data) {
            const size_t spec_off = ckpt.data_size + ckpt.dft_data_size;
            if (ckpt.spec_data_size > 0) {
                out_spec_data->assign(blob.begin() + spec_off,
                                      blob.begin() + spec_off + ckpt.spec_data_size);
            } else {
                out_spec_data->clear();
            }
        }
    };

    // Check hot cache
    auto hot_it = cache->hot_cache.find(checkpoint_id);
    if (hot_it != cache->hot_cache.end()) {
        auto& ckpt = cache->index[checkpoint_id];
        split_blob(hot_it->second, ckpt);
        ckpt.last_access = now_ms();
        ckpt.access_count++;
        cache->stats_hits++;
        return true;
    }

    // Check warm cache
    auto warm_it = cache->warm_cache.find(checkpoint_id);
    if (warm_it != cache->warm_cache.end()) {
        std::vector<uint8_t> uncached_blob;
        if (promote_to_hot(cache, checkpoint_id, &uncached_blob)) {
            auto hot = cache->hot_cache.find(checkpoint_id);
            auto & ckpt = cache->index[checkpoint_id];
            if (hot != cache->hot_cache.end()) {
                split_blob(hot->second, ckpt);
            } else if (!uncached_blob.empty()) {
                split_blob(uncached_blob, ckpt);
            } else {
                cache->stats_misses++;
                return false;
            }
            cache->stats_hits++;
            return true;
        }
        cache->stats_misses++;
        return false;
    }

    auto checkpoint_it = cache->index.find(checkpoint_id);
    if (checkpoint_it != cache->index.end()) {
        const auto & ckpt = checkpoint_it->second;
        if (ckpt.data_size <= SIZE_MAX - ckpt.dft_data_size &&
            ckpt.spec_data_size <= SIZE_MAX - ckpt.data_size - ckpt.dft_data_size &&
            ckpt.data_size + ckpt.dft_data_size + ckpt.spec_data_size > cache->config.hot_ram_bytes) {
            if (load_cold_direct(cache, checkpoint_id, out_data, out_dft_data, out_spec_data)) {
                cache->stats_hits++;
                return true;
            }
            cache->stats_misses++;
            return false;
        }
    }

    // Load from SSD. Oversized checkpoints are returned through a temporary
    // blob and remain cold instead of violating the hot RAM budget.
    std::vector<uint8_t> uncached_blob;
    if (promote_to_hot(cache, checkpoint_id, &uncached_blob)) {
        auto it = cache->hot_cache.find(checkpoint_id);
        if (it != cache->hot_cache.end()) {
            auto& ckpt = cache->index[checkpoint_id];
            split_blob(it->second, ckpt);
            cache->stats_hits++;
            return true;
        }
        if (!uncached_blob.empty()) {
            auto& ckpt = cache->index[checkpoint_id];
            split_blob(uncached_blob, ckpt);
            cache->stats_hits++;
            return true;
        }
    }

    cache->stats_misses++;
    return false;
}

uint64_t kv_ssd_find_match(kv_ssd_cache* cache,
                           const uint32_t* tokens, size_t tokens_size,
                           uint32_t current_turn,
                           uint64_t max_n_tokens,
                           int32_t n_past,
                           int32_t* out_lcp,
                           bool* out_partial,
                           bool allow_partial)
{
    (void)current_turn;  // unused - was for cross-conversation matching
    (void)n_past;        // unused - was for tiered search
    if (!cache || !cache->initialized || !tokens || tokens_size == 0) return 0;

    if (out_lcp)     *out_lcp = 0;
    if (out_partial) *out_partial = false;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Within-conversation search: find the checkpoint with the longest
    // common prefix that fits within the task.
    uint64_t best_id = 0;
    int32_t best_lcp = 0;
    uint32_t best_turn = 0;
    uint64_t best_n_tokens = 0;
    bool best_full = false;  // true = full hash+size match, false = partial LCP

    for (const auto& [id, ckpt] : cache->index) {
        if (cache->compat_hash != 0 && ckpt.compat_hash != cache->compat_hash) continue;

        // Compute longest common prefix with stored token prefix
        const size_t cmp_count = std::min(tokens_size, ckpt.token_prefix.size());
        int32_t lcp = 0;
        for (size_t i = 0; i < cmp_count; i++) {
            if (tokens[i] == ckpt.token_prefix[i]) lcp++;
            else break;
        }
        if (lcp == 0) continue;

        // Determine match quality:
        //   Full match: the entire stored token sequence matches the input
        //     (hash check passes AND input is long enough). The entire
        //     checkpoint state is valid.
        //   Partial LCP match: prefix matches for >= KV_SSD_TOKEN_PREFIX_MAX
        //     tokens but the full sequence hash differs (e.g., an agent trim
        //     modified middle tokens, so the stored hash no longer matches).
        //     The state is valid only up to the LCP; the caller caps n_past
        //     to the LCP and strips post-LCP attention KV.
        //     Skip the max_n_tokens filter for partial matches — we only use
        //     lcp tokens, not ckpt.n_tokens.
        const bool size_ok    = (tokens_size >= ckpt.n_tokens);
        // Short-circuit: don't read past tokens array when too short.
        const bool hash_match = size_ok &&
            (ckpt.token_hash == kv_ssd_hash_tokens(tokens, ckpt.n_tokens));

        if (hash_match && size_ok) {
            // Full match: apply standard max_n_tokens filter.
            if (max_n_tokens > 0 && ckpt.n_tokens > max_n_tokens) continue;
        } else if (allow_partial && lcp >= (int32_t)KV_SSD_TOKEN_PREFIX_MAX) {
            // Partial LCP match: accept, caller caps to LCP.
            // max_n_tokens filter skipped — lcp <= tokens_size by construction.
        } else {
            continue;
        }

        // Full hashes verify the complete saved prefix, not just the first
        // 4096 indexed tokens. Prefer the longest reusable state before age;
        // otherwise a recent short shared anchor masks a much longer match.
        int score = (hash_match && size_ok) ? 2 : 1;
        int best_score = best_full ? 2 : 1;
        const uint64_t reusable = score == 2 ? ckpt.n_tokens : (uint64_t)lcp;
        const uint64_t best_reusable = best_full ? best_n_tokens : (uint64_t)best_lcp;

        if (best_id == 0 ||
            score > best_score ||
            (score == best_score && reusable > best_reusable) ||
            (score == best_score && reusable == best_reusable && ckpt.turn_created > best_turn) ||
            (score == best_score && reusable == best_reusable && ckpt.turn_created == best_turn && id > best_id)) {
            best_full = (hash_match && size_ok);
            best_turn = ckpt.turn_created;
            best_n_tokens = ckpt.n_tokens;
            best_lcp = lcp;
            best_id = id;
        }
    }

    if (best_id != 0) {
        auto it = cache->index.find(best_id);
        uint64_t ntok = it != cache->index.end() ? (uint64_t)it->second.n_tokens : 0;
        LOG_INF("SSD cache: match checkpoint %lu conv=%016" PRIx64
                " turn=%u n_tokens=%lu lcp=%d%s\n",
                (unsigned long)best_id, cache->conv_hash,
                best_turn, (unsigned long)ntok, best_lcp,
                best_full ? "" : " (partial LCP)");
    }
    if (best_id != 0 && out_lcp)     *out_lcp = best_lcp;
    if (best_id != 0 && out_partial) *out_partial = !best_full;

    // Refresh only the small retention hint, not the large checkpoint file.
    auto & anchors = cache->prefix_checkpoint_ids;
    const auto anchor = std::find(anchors.begin(), anchors.end(), best_id);
    if (anchor != anchors.end() && std::next(anchor) != anchors.end()) {
        anchors.erase(anchor);
        anchors.push_back(best_id);
        write_index_file(cache);
    }

    return best_id;
}

uint64_t kv_ssd_find_by_slot(kv_ssd_cache* cache,
                             uint32_t slot_id,
                             uint64_t min_tokens,
                             uint32_t current_turn)
{
    if (!cache || !cache->initialized) return 0;

    std::lock_guard<std::mutex> lock(cache->mutex);

    uint64_t best_id = 0;
    uint64_t best_tokens = 0;

    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.slot_id != slot_id) continue;
        if (cache->compat_hash != 0 && ckpt.compat_hash != cache->compat_hash) continue;
        if (min_tokens > 0 && ckpt.n_tokens < min_tokens) continue;
        if (current_turn > 0 && ckpt.turn_id + (uint32_t)cache->config.warm_turns < current_turn) continue;
        if (ckpt.n_tokens > best_tokens) {
            best_tokens = ckpt.n_tokens;
            best_id = id;
        }
    }

    return best_id;
}

void kv_ssd_on_turn_complete(kv_ssd_cache* cache, uint32_t turn_id) {
    if (!cache || !cache->initialized) return;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Demote hot entries inactive for too many turns
    std::vector<uint64_t> to_demote_hw;
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.tier == KV_TIER_HOT && ckpt.turn_id + (uint32_t)cache->config.hot_turns <= turn_id) {
            to_demote_hw.push_back(id);
        }
    }
    for (uint64_t id : to_demote_hw) {
        auto it = cache->hot_cache.find(id);
        if (it != cache->hot_cache.end()) {
            size_t sz = it->second.size();
            cache->warm_cache[id] = std::move(it->second);
            cache->hot_cache.erase(it);
            cache->hot_bytes -= sz;
            cache->warm_bytes += sz;
            cache->index[id].tier = KV_TIER_WARM;
            cache->stats_evicts++;
        }
    }

    // Demote warm entries inactive for too many turns
    std::vector<uint64_t> to_demote_wc;
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.tier == KV_TIER_WARM && ckpt.turn_id + (uint32_t)cache->config.warm_turns <= turn_id) {
            to_demote_wc.push_back(id);
        }
    }
    for (uint64_t id : to_demote_wc) {
        auto it = cache->warm_cache.find(id);
        if (it != cache->warm_cache.end()) {
            size_t sz = it->second.size();
            cache->warm_cache.erase(it);
            cache->warm_bytes -= sz;
            cache->index[id].tier = KV_TIER_COLD;
            cache->stats_evicts++;
        }
    }

    // Ring buffer eviction: delete oldest cold checkpoints from disk
    ring_buffer_evict(cache);

    LOG_INF("SSD cache: turn %u complete (hot=%zu MiB warm=%zu MiB cold=%zu checkpoints=%zu)\n",
            turn_id,
            cache->hot_bytes / 1024 / 1024,
            cache->warm_bytes / 1024 / 1024,
            cache->index.size() - cache->hot_cache.size() - cache->warm_cache.size(),
            cache->index.size());
}

bool kv_ssd_get_meta(kv_ssd_cache* cache, uint64_t id, kv_ssd_checkpoint& out_meta) {
    if (!cache || !cache->initialized) return false;
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto it = cache->index.find(id);
    if (it == cache->index.end()) return false;
    out_meta = it->second;
    return true;
}

void kv_ssd_get_stats(kv_ssd_cache* cache,
                      size_t* out_hot_bytes, size_t* out_warm_bytes,
                      size_t* out_cold_count, size_t* out_total_count,
                      uint64_t* out_hits, uint64_t* out_misses)
{
    if (!cache) return;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (out_hot_bytes)  *out_hot_bytes  = cache->hot_bytes;
    if (out_warm_bytes) *out_warm_bytes = cache->warm_bytes;
    if (out_total_count) *out_total_count = cache->index.size();
    if (out_cold_count) {
        size_t cold = 0;
        for (const auto& [id, ckpt] : cache->index) {
            if (ckpt.tier == KV_TIER_COLD) cold++;
        }
        *out_cold_count = cold;
    }
    if (out_hits)   *out_hits   = cache->stats_hits;
    if (out_misses) *out_misses = cache->stats_misses;
}

uint32_t kv_ssd_get_max_turn_id(kv_ssd_cache* cache) {
    if (!cache || !cache->initialized) return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);

    uint32_t max_turn = 0;
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.turn_created > max_turn) {
            max_turn = ckpt.turn_created;
        }
    }
    return max_turn;
}

bool kv_ssd_forget_checkpoint(kv_ssd_cache* cache, uint64_t checkpoint_id) {
    if (!cache || checkpoint_id == 0) return false;
    std::lock_guard<std::mutex> lock(cache->mutex);
    return forget_checkpoint_locked(cache, checkpoint_id);
}

size_t kv_ssd_enforce_size_cap(
    const char* base_path,
    size_t max_bytes,
    std::vector<kv_ssd_evicted_checkpoint>* evicted) {
    namespace fs = std::filesystem;
    if (!base_path) return 0;

    struct disk_checkpoint {
        fs::path path;
        uint64_t cache_key = 0;
        uint64_t checkpoint_id = 0;
        size_t size_bytes = 0;
        int64_t modified_order = 0;
        bool user_scoped = false;
        bool protected_newest = false;
        bool prefix_anchor = false;
    };

    auto is_hex_key = [](const std::string& name) {
        return name.size() == 16 &&
            std::all_of(name.begin(), name.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
            });
    };
    auto checkpoint_id_from_name = [](const std::string& name, uint64_t& id) {
        if (name.size() < 10 || name.compare(0, 5, "ckpt-") != 0 ||
            name.compare(name.size() - 4, 4, ".bin") != 0) return false;
        const std::string number = name.substr(5, name.size() - 9);
        if (number.empty() || !std::all_of(number.begin(), number.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            })) return false;
        char* end = nullptr;
        id = std::strtoull(number.c_str(), &end, 10);
        return end && *end == '\0' && id != 0;
    };

    std::vector<disk_checkpoint> files;
    size_t total = 0;
    auto scan_namespace = [&](const fs::path& namespace_path, bool user_scoped) {
        std::error_code ec;
        if (!fs::is_directory(namespace_path, ec) || ec) return;
        fs::directory_iterator dirs(namespace_path, ec);
        const fs::directory_iterator end;
        for (; !ec && dirs != end; dirs.increment(ec)) {
            std::error_code dir_ec;
            if (!dirs->is_directory(dir_ec) || dir_ec) continue;
            const std::string dir_name = dirs->path().filename().string();
            if (!is_hex_key(dir_name)) continue;
            uint64_t cache_key = 0;
            if (sscanf(dir_name.c_str(), "%016" SCNx64, &cache_key) != 1) continue;

            // Index hints cover unloaded namespaces too. Invalid/missing
            // hints merely lose retention preference, never state validity.
            kv_ssd_index_header hdr = {};
            int index_fd = open((dirs->path() / "index.bin").string().c_str(), O_RDONLY);
            const bool hints_valid = index_fd >= 0 && pread_all(index_fd, &hdr, sizeof(hdr), 0) &&
                hdr.magic == KV_SSD_MAGIC_INDEX && hdr.version == KV_SSD_VERSION &&
                hdr.header_checksum != 0 && validate_checksum(hdr, "retention index");
            if (index_fd >= 0) close(index_fd);

            std::error_code file_ec;
            fs::directory_iterator entries(dirs->path(), file_ec);
            for (; !file_ec && entries != end; entries.increment(file_ec)) {
                std::error_code regular_ec;
                if (!entries->is_regular_file(regular_ec) || regular_ec) continue;
                uint64_t checkpoint_id = 0;
                if (!checkpoint_id_from_name(entries->path().filename().string(), checkpoint_id)) continue;
                std::error_code size_ec;
                const uintmax_t raw_size = entries->file_size(size_ec);
                if (size_ec || raw_size > SIZE_MAX) continue;
                std::error_code time_ec;
                const auto modified = fs::last_write_time(entries->path(), time_ec);
                const int64_t modified_order = time_ec ? 0 :
                    (int64_t)modified.time_since_epoch().count();
                const size_t file_size = (size_t) raw_size;
                files.push_back({ entries->path(), cache_key, checkpoint_id,
                                  file_size, modified_order, user_scoped, false,
                                  hints_valid && std::find(std::begin(hdr.prefix_checkpoint_ids),
                                      std::end(hdr.prefix_checkpoint_ids), checkpoint_id) !=
                                      std::end(hdr.prefix_checkpoint_ids) });
                total = file_size > SIZE_MAX - total ? SIZE_MAX : total + file_size;
            }
        }
    };

    const fs::path base(base_path);
    scan_namespace(base, false);
    scan_namespace(base / "u", true);
    if (max_bytes == 0 || total <= max_bytes) return total;

    // Prefer to retain the newest checkpoint in each conversation. If those
    // checkpoints alone exceed the cap, evict the oldest conversations too;
    // the newest checkpoint overall remains protected so the cache retains at
    // least one recovery point whenever a single checkpoint fits the cap.
    std::unordered_map<std::string, size_t> newest_by_directory;
    for (size_t i = 0; i < files.size(); ++i) {
        const std::string directory = files[i].path.parent_path().string();
        auto it = newest_by_directory.find(directory);
        if (it == newest_by_directory.end()) {
            newest_by_directory[directory] = i;
            continue;
        }
        const auto& current = files[it->second];
        if (files[i].modified_order > current.modified_order ||
            (files[i].modified_order == current.modified_order &&
             files[i].checkpoint_id > current.checkpoint_id)) {
            it->second = i;
        }
    }
    for (const auto& [directory, index] : newest_by_directory) {
        (void)directory;
        files[index].protected_newest = true;
    }

    size_t newest_overall = 0;
    for (size_t i = 1; i < files.size(); ++i) {
        if (files[i].modified_order > files[newest_overall].modified_order ||
            (files[i].modified_order == files[newest_overall].modified_order &&
             files[i].checkpoint_id > files[newest_overall].checkpoint_id)) {
            newest_overall = i;
        }
    }

    std::vector<size_t> candidates;
    for (size_t i = 0; i < files.size(); ++i) {
        if (!files[i].protected_newest) candidates.push_back(i);
    }
    auto older_first = [&](size_t a, size_t b) {
        if (files[a].modified_order != files[b].modified_order) {
            return files[a].modified_order < files[b].modified_order;
        }
        if (files[a].cache_key != files[b].cache_key) {
            return files[a].cache_key < files[b].cache_key;
        }
        return files[a].checkpoint_id < files[b].checkpoint_id;
    };
    std::sort(candidates.begin(), candidates.end(), older_first);

    std::vector<size_t> oldest_conversations;
    for (size_t i = 0; i < files.size(); ++i) {
        if (files[i].protected_newest && i != newest_overall) oldest_conversations.push_back(i);
    }
    std::sort(oldest_conversations.begin(), oldest_conversations.end(), older_first);
    candidates.insert(candidates.end(), oldest_conversations.begin(), oldest_conversations.end());
    // Prefer shared anchors after ordinary branch checkpoints. They remain
    // eligible victims when necessary to meet the global disk budget.
    std::stable_partition(candidates.begin(), candidates.end(),
        [&](size_t i) { return !files[i].prefix_anchor; });

    for (size_t index : candidates) {
        if (total <= max_bytes) break;
        const auto& file = files[index];
        std::error_code remove_ec;
        fs::remove(file.path, remove_ec);
        if (remove_ec) continue;
        total = file.size_bytes > total ? 0 : total - file.size_bytes;
        if (evicted) {
            evicted->push_back({ file.cache_key, file.checkpoint_id,
                                 file.size_bytes, file.user_scoped });
        }
        LOG_INF("SSD cache: global cap evicted %s checkpoint key=%016" PRIx64
                " id=%lu size=%zu MiB remaining=%zu MiB cap=%zu MiB\n",
                file.user_scoped ? "user" : "anonymous", file.cache_key,
                (unsigned long)file.checkpoint_id, file.size_bytes / 1024 / 1024,
                total / 1024 / 1024, max_bytes / 1024 / 1024);
    }

    if (total > max_bytes) {
        LOG_WRN("SSD cache: global cap remains above target (%zu MiB > %zu MiB) because the newest checkpoint is protected\n",
                total / 1024 / 1024, max_bytes / 1024 / 1024);
    }
    return total;
}

// =============================================================================
// Conversation continuation and global operations
// =============================================================================

// Scan all conversation directories for a fuzzy prefix match.
// Used for conversation continuation detection after restart.
// Returns conv_hash of best match, or 0 if none found above min_overlap.
uint64_t kv_ssd_find_continuation(
    const char* base_path,
    const uint32_t* tokens, size_t tokens_size,
    float min_overlap,
    uint64_t compat_hash,
    float* out_overlap)
{
    if (!base_path || !tokens || tokens_size == 0) return 0;

    if (out_overlap) *out_overlap = 0.0f;

    namespace fs = std::filesystem;
    fs::path base(base_path);
    std::error_code dir_ec;
    if (!fs::is_directory(base, dir_ec) || dir_ec) return 0;

    uint64_t best_conv = 0;
    float best_score = 0.0f;

    fs::directory_iterator entry_it(base, dir_ec);
    const fs::directory_iterator end;
    for (; !dir_ec && entry_it != end; entry_it.increment(dir_ec)) {
        const auto& entry = *entry_it;
        std::error_code entry_ec;
        if (!entry.is_directory(entry_ec) || entry_ec) continue;
        const std::string dirname = entry.path().filename().string();
        if (dirname.size() != 16 ||
            !std::all_of(dirname.begin(), dirname.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
            })) {
            continue;
        }

        std::string conv_dir = entry.path().string();

        // Parse conv_hash from directory name
        uint64_t conv_hash = 0;
        if (sscanf(dirname.c_str(), "%016" SCNx64, &conv_hash) != 1) continue;

        // Load the index file for this conversation
        std::string index_file = conv_dir + "/index.bin";
        int fd = open(index_file.c_str(), O_RDONLY);
        if (fd < 0) continue;

        kv_ssd_index_header hdr;
        bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
        close(fd);
        if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX) continue;
        // Only scan caches whose on-disk version matches the running
        // version. Older versions are a clean miss; newer versions
        // imply an upgraded-downgraded binary mismatch and are also
        // rejected.
        if (hdr.version != KV_SSD_VERSION) continue;

        // Skip directories with mismatched model config
        if (compat_hash != 0 && hdr.compat_hash != 0 && hdr.compat_hash != compat_hash) continue;

        // Scan for checkpoint with best prefix overlap
        float best_conv_score = 0.0f;

        std::error_code ckpt_ec;
        fs::directory_iterator ckpt_it(entry.path(), ckpt_ec);
        for (; !ckpt_ec && ckpt_it != end; ckpt_it.increment(ckpt_ec)) {
            const auto& ckpt_entry = *ckpt_it;
            const std::string fname = ckpt_entry.path().filename().string();
            if (fname.size() < 9) continue;
            if (fname.compare(0, 5, "ckpt-") != 0) continue;
            if (fname.compare(fname.size() - 4, 4, ".bin") != 0) continue;

            std::string ckpt_file = ckpt_entry.path().string();
            int cfd = open(ckpt_file.c_str(), O_RDONLY);
            if (cfd < 0) continue;

            kv_ssd_record rec;
            bool rok = pread_all(cfd, &rec, sizeof(rec), 0);
            close(cfd);
            if (!rok || rec.magic != KV_SSD_MAGIC_REC || rec.version != KV_SSD_VERSION) continue;

            // Compute overlap with stored prefix
            size_t cmp_count = std::min(tokens_size, (size_t)rec.token_count);
            if (cmp_count < 16) continue; // too short to be meaningful

            size_t matches = 0;
            for (size_t i = 0; i < cmp_count; i++) {
                if (tokens[i] == rec.token_prefix[i]) matches++;
                else break;
            }

            // Score: how much of the checkpoint's verifiable prefix we matched.
            // matches is bounded by KV_SSD_TOKEN_PREFIX_MAX (4096) - that's all
            // the prefix we ever store. Normalizing by n_tokens (the full state
            // extent) made any checkpoint past ~4551 tokens invisible: with the
            // default 0.90 min_overlap, matches/n_tokens caps at 4096/n_tokens
            // and never reaches threshold. Normalize by token_count instead -
            // a full-prefix match scores 1.0 regardless of total depth.
            // token_count <= KV_SSD_TOKEN_PREFIX_MAX always (cap in kv_ssd_store),
            // so the ratio is bounded by 1.0.
            float score = (rec.token_count > 0)
                ? std::min(1.0f, (float)matches / (float)rec.token_count)
                : 0.0f;
            if (score > best_conv_score) best_conv_score = score;
        }

        if (best_conv_score >= min_overlap && best_conv_score > best_score) {
            best_score = best_conv_score;
            best_conv = conv_hash;
        }
    }

    if (best_conv != 0) {
        LOG_INF("SSD cache: continuation found conv=%016" PRIx64 " overlap=%.1f%%\n",
                best_conv, best_score * 100.0f);
        if (out_overlap) *out_overlap = best_score;
    }

    return best_conv;
}

// Get maximum turn_id across all anonymous and user-scoped conversation
// directories. User caches live one level lower under "u/"; omitting them
// resets the server's turn counter after restart and distorts tier aging and
// oldest-first eviction for the workloads that use stable user IDs.
uint32_t kv_ssd_get_max_turn_id_global(const char* base_path) {
    if (!base_path) return 0;

    namespace fs = std::filesystem;
    fs::path base(base_path);
    std::error_code ec;
    if (!fs::is_directory(base, ec) || ec) return 0;

    uint32_t max_turn = 0;

    auto scan_namespace = [&](const fs::path& namespace_dir) {
        std::error_code dir_ec;
        if (!fs::is_directory(namespace_dir, dir_ec) || dir_ec) return;

        fs::directory_iterator entry_it(namespace_dir, dir_ec);
        const fs::directory_iterator end;
        for (; !dir_ec && entry_it != end; entry_it.increment(dir_ec)) {
            const auto& entry = *entry_it;
            std::error_code entry_ec;
            if (!entry.is_directory(entry_ec) || entry_ec) continue;
            const std::string dirname = entry.path().filename().string();
            if (dirname.size() != 16 ||
                !std::all_of(dirname.begin(), dirname.end(), [](unsigned char c) {
                    return std::isxdigit(c) != 0;
                })) {
                continue;
            }

            const std::string conv_dir = entry.path().string();
            const std::string index_file = conv_dir + "/index.bin";
            int fd = open(index_file.c_str(), O_RDONLY);
            if (fd < 0) continue;

            kv_ssd_index_header hdr;
            bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
            close(fd);
            if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX || hdr.version != KV_SSD_VERSION) continue;

            std::error_code ckpt_ec;
            fs::directory_iterator ckpt_it(entry.path(), ckpt_ec);
            for (; !ckpt_ec && ckpt_it != end; ckpt_it.increment(ckpt_ec)) {
                const auto& ckpt_entry = *ckpt_it;
                const std::string fname = ckpt_entry.path().filename().string();
                if (fname.size() < 9 ||
                    fname.compare(0, 5, "ckpt-") != 0 ||
                    fname.compare(fname.size() - 4, 4, ".bin") != 0) {
                    continue;
                }

                int cfd = open(ckpt_entry.path().c_str(), O_RDONLY);
                if (cfd < 0) continue;

                kv_ssd_record rec;
                bool rok = pread_all(cfd, &rec, sizeof(rec), 0);
                close(cfd);
                if (rok && rec.magic == KV_SSD_MAGIC_REC &&
                    rec.version == KV_SSD_VERSION &&
                    rec.turn_created > max_turn) {
                    max_turn = rec.turn_created;
                }
            }
        }
    };

    scan_namespace(base);
    scan_namespace(base / "u");

    return max_turn;
}

void kv_ssd_prefetch(kv_ssd_cache* cache, uint64_t checkpoint_id) {
    if (!cache || !cache->initialized || checkpoint_id == 0) return;
    ckpt_readahead(cache, checkpoint_id);
}

void kv_ssd_prefetch_slot(kv_ssd_cache* cache, uint32_t slot_id) {
    if (!cache || !cache->initialized) return;

    // Prefetch all cold checkpoints for this slot
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.slot_id == slot_id && ckpt.tier == KV_TIER_COLD) {
            ckpt_readahead(cache, id);
        }
    }
}
