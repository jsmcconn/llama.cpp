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
#include <climits>
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
uint64_t kv_ssd_hash_tokens(const uint32_t* tokens, size_t count) {
    if (!tokens || count == 0) return 0;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < count; i++) {
        uint32_t v = tokens[i];
        h ^= (uint64_t)(v & 0xFF);
        h *= 1099511628211ULL;
        h ^= (uint64_t)((v >> 8) & 0xFF);
        h *= 1099511628211ULL;
        h ^= (uint64_t)((v >> 16) & 0xFF);
        h *= 1099511628211ULL;
        h ^= (uint64_t)((v >> 24) & 0xFF);
        h *= 1099511628211ULL;
    }
    return h;
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
    if (rec.version != KV_SSD_VERSION) return false;
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
    c->slot_latest[rec.slot_id] = rec.id;
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
        [](const auto& a, const auto& b) { return a.second < b.second; });

    int to_evict = (int)cold_count - max_cold;
    int evicted = 0;
    for (int i = 0; i < to_evict && i < (int)cold_by_age.size(); i++) {
        uint64_t id = cold_by_age[i].first;
        auto ckpt_it = c->index.find(id);
        if (ckpt_it == c->index.end()) continue;

        uint32_t slot = ckpt_it->second.slot_id;

        // Remove from slot_latest if this was the latest for that slot
        auto slot_it = c->slot_latest.find(slot);
        if (slot_it != c->slot_latest.end() && slot_it->second == id) {
            c->slot_latest.erase(slot_it);
        }

        // Delete file from disk
        delete_checkpoint_file(c, id);

        // Remove from in-memory index
        c->index.erase(ckpt_it);
        evicted++;
    }

    if (evicted > 0) {
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
static bool promote_to_hot(kv_ssd_cache* c, uint64_t id) {
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
        c->index.erase(id);
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
    if (!validate_checksum(rec, "checkpoint")) return false;
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

    const size_t tgt_size  = (size_t)rec.data_size;
    const size_t dft_size  = (size_t)rec.dft_data_size;
    const size_t spec_size = (size_t)rec.spec_data_size;
    const size_t total_blob = tgt_size + dft_size + spec_size;
    make_room_hot(c, total_blob);

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

    c->hot_cache[id] = std::move(data);
    c->hot_bytes += total_blob;
    ckpt.tier = KV_TIER_HOT;
    ckpt.last_access = now_ms();
    ckpt.access_count++;
    c->stats_loads++;

    LOG_INF("SSD cache: promoted checkpoint %lu cold->hot (%zu MiB)\n",
            (unsigned long)id, total_blob / 1024 / 1024);
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

    // Build combined blob: [tgt_data][dft_data][spec_data]
    const size_t total_blob = data_size + rec.dft_data_size + rec.spec_data_size;
    make_room_hot(cache, total_blob);

    // Store combined blob in hot cache
    std::vector<uint8_t> hot_blob;
    hot_blob.reserve(total_blob);
    hot_blob.assign(data, data + data_size);
    if (rec.dft_data_size > 0) hot_blob.insert(hot_blob.end(), dft_data, dft_data + rec.dft_data_size);
    if (rec.spec_data_size > 0) hot_blob.insert(hot_blob.end(), spec_data, spec_data + rec.spec_data_size);
    cache->hot_cache[id] = std::move(hot_blob);
    cache->hot_bytes += total_blob;

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
    ckpt.tier = KV_TIER_HOT;
    if (token_count > 0) {
        ckpt.token_prefix.assign(rec.token_prefix, rec.token_prefix + token_count);
    }
    ckpt.data_size      = data_size;
    ckpt.dft_data_size  = rec.dft_data_size;
    ckpt.spec_data_size = rec.spec_data_size;
    ckpt.last_access = now_ms();
    ckpt.access_count = 1;

    cache->index[id] = ckpt;
    cache->slot_latest[slot_id] = id;
    cache->stats_stores++;

    LOG_INF("SSD cache: stored checkpoint %lu slot=%u tokens=%lu tgt=%zu dft=%llu spec=%llu MiB "
            "(hot=%zu MiB warm=%zu MiB total=%zu)\n",
            (unsigned long)id, slot_id, (unsigned long)n_tokens,
            data_size / 1024 / 1024,
            (unsigned long long)(rec.dft_data_size / 1024 / 1024),
            (unsigned long long)(rec.spec_data_size / 1024 / 1024),
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
        auto& ckpt = cache->index[checkpoint_id];
        split_blob(warm_it->second, ckpt);
        promote_to_hot(cache, checkpoint_id);
        cache->stats_hits++;
        return true;
    }

    // Load from SSD file and promote to hot
    if (promote_to_hot(cache, checkpoint_id)) {
        auto it = cache->hot_cache.find(checkpoint_id);
        if (it != cache->hot_cache.end()) {
            auto& ckpt = cache->index[checkpoint_id];
            split_blob(it->second, ckpt);
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
                           bool* out_partial)
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
        } else if (lcp >= (int32_t)KV_SSD_TOKEN_PREFIX_MAX) {
            // Partial LCP match: accept, caller caps to LCP.
            // max_n_tokens filter skipped — lcp <= tokens_size by construction.
        } else {
            continue;
        }

        // Prefer full matches over partial matches, then highest LCP,
        // then most recent turn, then more tokens.
        int score = (hash_match && size_ok) ? 2 : 1;
        int best_score = best_full ? 2 : 1;

        if (best_id == 0 ||
            score > best_score ||
            (score == best_score && lcp > best_lcp) ||
            (score == best_score && lcp == best_lcp && ckpt.turn_created > best_turn) ||
            (score == best_score && lcp == best_lcp && ckpt.turn_created == best_turn && ckpt.n_tokens > best_n_tokens)) {
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

const kv_ssd_checkpoint* kv_ssd_get_meta(kv_ssd_cache* cache, uint64_t id) {
    if (!cache || !cache->initialized) return nullptr;
    auto it = cache->index.find(id);
    if (it == cache->index.end()) return nullptr;
    return &it->second;
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
    if (!fs::exists(base) || !fs::is_directory(base)) return 0;

    uint64_t best_conv = 0;
    float best_score = 0.0f;

    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        if (dirname[0] == '.') continue;

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

        for (const auto& ckpt_entry : fs::directory_iterator(conv_dir)) {
            std::string fname = ckpt_entry.path().filename().string();
            if (fname.size() < 9) continue;
            if (fname.compare(0, 5, "ckpt-") != 0) continue;
            if (fname.compare(fname.size() - 4, 4, ".bin") != 0) continue;

            std::string ckpt_file = ckpt_entry.path().string();
            int cfd = open(ckpt_file.c_str(), O_RDONLY);
            if (cfd < 0) continue;

            kv_ssd_record rec;
            bool rok = pread_all(cfd, &rec, sizeof(rec), 0);
            close(cfd);
            if (!rok || rec.magic != KV_SSD_MAGIC_REC) continue;

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

// Get maximum turn_id across all conversation directories.
uint32_t kv_ssd_get_max_turn_id_global(const char* base_path) {
    if (!base_path) return 0;

    namespace fs = std::filesystem;
    fs::path base(base_path);
    if (!fs::exists(base) || !fs::is_directory(base)) return 0;

    uint32_t max_turn = 0;

    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        if (dirname[0] == '.') continue;

        std::string conv_dir = entry.path().string();

        // Parse conv_hash (validate 16-char hex name)
        uint64_t conv_hash_test = 0;
        if (sscanf(dirname.c_str(), "%016" SCNx64, &conv_hash_test) != 1) continue;

        std::string index_file = conv_dir + "/index.bin";
        int fd = open(index_file.c_str(), O_RDONLY);
        if (fd < 0) continue;

        kv_ssd_index_header hdr;
        bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
        close(fd);
        if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX) continue;

        // Quick scan of checkpoint files for max turn_created
        for (const auto& ckpt_entry : fs::directory_iterator(conv_dir)) {
            std::string fname = ckpt_entry.path().filename().string();
            if (fname.size() < 9) continue;
            if (fname.compare(0, 5, "ckpt-") != 0) continue;
            if (fname.compare(fname.size() - 4, 4, ".bin") != 0) continue;

            std::string ckpt_file = ckpt_entry.path().string();
            int cfd = open(ckpt_file.c_str(), O_RDONLY);
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
