// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// SSD-Backed KV Cache with Hot/Warm/Cold Tiering
// Per-checkpoint file storage with ring buffer eviction.

#ifndef KV_SSD_CACHE_H
#define KV_SSD_CACHE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

// Tier states for cached checkpoints
enum kv_ssd_tier {
    KV_TIER_COLD  = 0,  // On SSD only
    KV_TIER_WARM  = 1,  // In RAM, recently used
    KV_TIER_HOT   = 2,  // In RAM, actively used
};

// FNV-1a hash for token sequences (exposed for use by server layer)
uint64_t kv_ssd_hash_tokens(const uint32_t* tokens, size_t count);

// Configuration for the SSD cache
struct kv_ssd_config {
    size_t hot_ram_bytes     = 2ULL * 1024 * 1024 * 1024;  // 2 GB hot tier
    size_t warm_ram_bytes    = 1ULL * 1024 * 1024 * 1024;  // 1 GB warm tier
    int hot_turns            = 2;   // Demote hot->warm after N turns inactive
    int warm_turns           = 4;   // Demote warm->cold after N turns inactive
    size_t hot_window_tokens = 4096; // Recent tokens always kept hot
    bool auto_size           = true;
    int max_cold_checkpoints = 32;  // Max checkpoints per model (ring buffer cap)
    int prefix_checkpoints  = 0;   // Prefer retaining up to 3 shared-prefix anchors within existing caps
    float memory_reserve     = 0.15f;
    // Size of the loaded model in bytes. The auto-size path uses this
    // to subtract the model's footprint from MemAvailable before
    // computing hot/warm budgets. On UMA APUs (Ayaneo Flip 7840U,
    // Strix Halo) the GTT-mapped model lives in the same system RAM
    // that the SSD cache tiers consume, and MemAvailable counts GTT
    // pages as evictable. Without this subtraction, auto-size reserves
    // ~75% of MemAvailable for the hot tier, which combined with a
    // 20+ GiB GTT-mapped model OOM-kills the server. Set to 0 to
    // preserve the legacy behavior (auto-size uses raw MemAvailable).
    size_t model_size_bytes  = 0;
    bool no_fsync            = false; // Skip fsync on write (faster, may lose last checkpoint on crash)
    // Durable-write cadence. Both disabled preserves the legacy behavior of
    // writing every checkpoint. When enabled, a checkpoint is durable when
    // either the compatible-prefix growth or wall-clock age reaches its cap.
    size_t durable_min_growth_tokens = 0;
    uint64_t durable_max_age_ms      = 0;
    // v4: model identity fields written to the index header and every
    // per-checkpoint header. Used to reject caches whose underlying
    // model no longer matches the on-disk metadata. Set both to 0 to
    // skip validation (legacy behavior).
    uint64_t model_identity  = 0; // hash(arch, dims, cache types)
    uint64_t model_hash      = 0; // content hash of the GGUF
    uint32_t quantization    = 0; // ggml_type of the primary weight tensor
};

enum kv_ssd_store_action {
    KV_SSD_STORE_WRITE = 0,
    KV_SSD_STORE_REUSE,
    KV_SSD_STORE_SKIP_CADENCE,
};

struct kv_ssd_store_plan {
    kv_ssd_store_action action = KV_SSD_STORE_WRITE;
    uint64_t checkpoint_id = 0;
    uint64_t base_n_tokens = 0;
    uint64_t growth_tokens = 0;
    uint64_t age_ms = 0;
};

struct kv_ssd_evicted_checkpoint {
    uint64_t cache_key = 0;
    uint64_t checkpoint_id = 0;
    size_t size_bytes = 0;
    bool user_scoped = false;
};

// Checkpoint metadata (in-memory index entry)
struct kv_ssd_checkpoint {
    uint64_t id;            // Unique checkpoint ID
    uint32_t slot_id;       // Originating slot
    int32_t pos_min;        // Position range min
    int32_t pos_max;        // Position range max
    uint64_t n_tokens;      // Token count covered
    uint32_t turn_id;       // Last turn accessed
    uint32_t turn_created;  // Turn when created
    uint64_t token_hash;    // FNV-1a hash of token sequence
    uint64_t compat_hash;   // Model config hash (arch, dims, cache types)
    size_t token_count;     // Tokens in stored prefix
    kv_ssd_tier tier;       // Current tier
    size_t data_size;       // Serialized tgt context size in bytes
    size_t dft_data_size;   // Serialized MTP/draft context size (0 = none)
    size_t spec_data_size;  // Serialized speculative impl state size (0 = none)
    uint64_t last_access;   // Timestamp ms of last access
    uint32_t access_count;  // Number of times accessed
    std::vector<uint32_t> token_prefix; // First N tokens for matching (cached in RAM)
    // v4: per-checkpoint copy of the cache's model identity. Allows
    // rejecting a checkpoint whose model no longer matches the active
    // model, even if the index file is intact (e.g. model swap).
    uint64_t model_identity = 0;
    uint32_t quantization   = 0;
    uint64_t model_hash     = 0;
};

// Index file header (written to {MODEL}/index.bin)
struct kv_ssd_index_header {
    uint32_t magic;         // "KVID" = 0x4B564944
    uint32_t version;       // Format version
    uint64_t next_id;       // Next checkpoint ID to allocate
    uint64_t compat_hash;   // Model compatibility hash
    // v4 fields. `reserved[]` was a generic future-use slot in v3; in
    // v4 the first 9 of the 12 slots are now defined. Readers must
    // validate magic + version (== KV_SSD_VERSION) before reading any
    // of these.
    uint32_t cache_format_version;  // KV_SSD_CACHE_FORMAT_VERSION
    uint32_t quantization;          // GGUF quantization (ggml_type)
    uint64_t model_identity;        // hash(arch, dims, cache types) -
                                    // identifies the model *family*
                                    // independent of compat_hash
    uint64_t model_hash;            // content hash of the GGUF file
                                    // (or 0 if not computed)
    uint64_t payload_size;          // bytes in the index file after
                                    // the header (currently 0 - index
                                    // is a fixed-size record; reserved
                                    // for future per-cache metadata)
    uint64_t header_checksum;       // FNV-1a over the prior header
                                    // bytes; detects on-disk corruption
    // Formerly reserved in v4. Optional retention hints, oldest -> newest.
    // They do not change checkpoint validity; older v4 readers ignore them.
    uint64_t prefix_checkpoint_ids[3];
};

// Maximum tokens stored per checkpoint for prefix matching
#define KV_SSD_TOKEN_PREFIX_MAX 4096

// Per-checkpoint file header (fixed size, followed by checkpoint data)
// v4 layout: [kv_ssd_record][tgt_data][dft_data][spec_data]
// data_size = tgt bytes; dft_data_size/spec_data_size = optional extra blobs (0 = absent)
struct kv_ssd_record {
    uint32_t magic;         // KV_SSD_MAGIC_REC
    uint32_t version;
    uint64_t id;
    uint32_t slot_id;
    int32_t pos_min;
    int32_t pos_max;
    uint64_t n_tokens;
    uint32_t turn_created;
    uint64_t data_size;     // tgt context bytes following this header
    uint64_t token_hash;    // Hash of full token sequence
    uint32_t token_count;   // Tokens stored in prefix array
    uint64_t compat_hash;   // Model config hash (arch, dims, cache types)
    uint32_t token_prefix[KV_SSD_TOKEN_PREFIX_MAX]; // First N tokens
    uint64_t dft_data_size; // MTP/draft context bytes appended after tgt_data (0 = none)
    uint64_t spec_data_size;// Speculative impl state bytes appended after dft_data (0 = none)
    // v4 metadata (mirrors kv_ssd_index_header). Carried per-checkpoint
    // so an individual checkpoint can be rejected even if the index
    // is intact (e.g. when restoring after a model file swap).
    uint32_t cache_format_version;
    uint32_t quantization;
    uint64_t model_identity;
    uint64_t model_hash;
    uint64_t payload_size;      // bytes after this header in this file
    uint64_t header_checksum;   // FNV-1a over the prior header bytes
};

class kv_ssd_cache {
public:
    kv_ssd_cache() = default;
    ~kv_ssd_cache() = default;

    kv_ssd_cache(const kv_ssd_cache&) = delete;
    kv_ssd_cache& operator=(const kv_ssd_cache&) = delete;

    // Directory state
    std::string base_path;          // e.g. ssd-cache/
    std::string model_dir;          // e.g. ssd-cache/{CONV_HASH}/
    uint64_t conv_hash = 0;         // This cache's conversation identity
    // v4: cached copy of the v4 metadata fields. Populated from
    // kv_ssd_config on init, then refreshed from the on-disk index
    // header on read_index_file. Used as the source of truth for
    // per-checkpoint header writes and for downstream validation.
    uint64_t model_identity = 0;
    uint32_t quantization   = 0;
    uint64_t model_hash     = 0;
    kv_ssd_config config;
    bool initialized = false;

    // Checkpoint index: id -> metadata
    std::unordered_map<uint64_t, kv_ssd_checkpoint> index;
    std::vector<uint64_t> prefix_checkpoint_ids; // bounded retention hints, oldest -> newest

    // RAM caches: id -> checkpoint data
    std::unordered_map<uint64_t, std::vector<uint8_t>> hot_cache;
    std::unordered_map<uint64_t, std::vector<uint8_t>> warm_cache;

    // RAM usage tracking
    size_t hot_bytes  = 0;
    size_t warm_bytes = 0;

    // Next checkpoint ID
    uint64_t next_id = 1;

    // Stats
    uint64_t stats_hits    = 0;
    uint64_t stats_misses  = 0;
    uint64_t stats_stores  = 0;
    uint64_t stats_evicts  = 0;
    uint64_t stats_loads   = 0;

    // Mutex for thread safety
    std::mutex mutex;

    // Model compatibility hash for config validation on load.
    // Checkpoints with mismatched compat_hash are rejected.
    uint64_t compat_hash = 0;

    // Reverse lookup: slot_id -> most recent checkpoint id
    std::unordered_map<uint32_t, uint64_t> slot_latest;
};

// =============================================================================
// Public API
// =============================================================================

// Initialize the SSD cache. Creates {path}/{conv_hash}/ directory.
// Returns nullptr on failure.
// namespace_prefix lets callers segregate cache directories under a subpath.
// Empty (default) keeps the legacy layout: {path}/{hash_hex}/.
// Pass "u/" to route to {path}/u/{hash_hex}/, which isolates user_id caches
// from anonymous conv_hash caches on disk.
kv_ssd_cache* kv_ssd_init(const char* path, const kv_ssd_config* cfg, uint64_t conv_hash, const char* namespace_prefix = "");

// Shutdown: write index and cleanup.
void kv_ssd_free(kv_ssd_cache* cache);

// Store a checkpoint. Written to ckpt-{next_id}.bin immediately, kept in hot tier.
// tokens/tokens_size used for hash-based matching (can be null/0).
// dft_data/spec_data are optional extra blobs (MTP context and speculative impl state).
// Returns checkpoint ID (>0) on success, 0 on failure.
uint64_t kv_ssd_store(kv_ssd_cache* cache,
                  uint32_t slot_id,
                  const uint8_t* data, size_t data_size,
                  int32_t pos_min, int32_t pos_max,
                  uint64_t n_tokens, uint32_t turn_id,
                  const uint32_t* tokens, size_t tokens_size,
                  uint64_t compat_hash = 0,
                  const uint8_t* dft_data = nullptr, size_t dft_data_size = 0,
                  const uint8_t* spec_data = nullptr, size_t spec_data_size = 0);

// Decide whether a durable checkpoint should be serialized and written.
// Exact duplicate prompts are reused without serialization. Compatible
// append-only prompts are rate-limited by the configured growth/time cadence.
// Branches whose prior durable state is not an exact prefix always write.
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
    bool prefix_anchor = false);

// Prefer retaining an existing checkpoint as a shared-prefix anchor. Hints
// survive restart in the v4 index, but never override cold-count/disk caps.
void kv_ssd_mark_prefix(kv_ssd_cache* cache, uint64_t checkpoint_id);

// Load a checkpoint by ID. Reads ckpt-{id}.bin and promotes to hot tier.
// out_dft_data and out_spec_data receive the optional extra blobs if non-null.
// Returns true and copies data to out_data on success.
bool kv_ssd_load(kv_ssd_cache* cache,
                 uint64_t checkpoint_id,
                 std::vector<uint8_t>& out_data,
                 std::vector<uint8_t>* out_dft_data = nullptr,
                 std::vector<uint8_t>* out_spec_data = nullptr);

// Find best matching checkpoint by token prefix comparison.
// Searches within this conversation's cache only.
// Returns checkpoint ID or 0 if no match found.
uint64_t kv_ssd_find_match(kv_ssd_cache* cache,
                           const uint32_t* tokens, size_t tokens_size,
                           uint32_t current_turn,
                           uint64_t max_n_tokens,
                           int32_t n_past = -1,
                           int32_t* out_lcp = nullptr,
                           bool* out_partial = nullptr,
                           bool allow_partial = true);

// Find best checkpoint for a slot.
// Returns checkpoint ID or 0 if none found.
uint64_t kv_ssd_find_by_slot(kv_ssd_cache* cache,
                             uint32_t slot_id,
                             uint64_t min_tokens,
                             uint32_t current_turn);

// Notify turn completion. Triggers tier demotion and ring buffer pruning.
void kv_ssd_on_turn_complete(kv_ssd_cache* cache, uint32_t turn_id);

// Copy checkpoint metadata while holding the cache lock.
bool kv_ssd_get_meta(kv_ssd_cache* cache, uint64_t id, kv_ssd_checkpoint& out_meta);

// Set model compatibility hash for config validation on load.
void kv_ssd_set_compat_hash(kv_ssd_cache* cache, uint64_t compat_hash);

// Get stats.
void kv_ssd_get_stats(kv_ssd_cache* cache,
                      size_t* hot_bytes, size_t* warm_bytes,
                      size_t* cold_count, size_t* total_count,
                      uint64_t* hits, uint64_t* misses);
// Get the maximum turn_id across all existing checkpoints.
uint32_t kv_ssd_get_max_turn_id(kv_ssd_cache* cache);

// Remove an entry from an already loaded cache after a global filesystem
// eviction. The checkpoint file is assumed to have been removed already.
bool kv_ssd_forget_checkpoint(kv_ssd_cache* cache, uint64_t checkpoint_id);

// Enforce a real on-disk byte cap across anonymous and user-scoped caches.
// Every ckpt-*.bin file counts regardless of its in-process tier. Eviction is
// oldest-checkpoint-first while preserving the newest file in each directory.
// Returns the number of bytes remaining after eviction and reports removed
// loaded-cache identities through `evicted`.
size_t kv_ssd_enforce_size_cap(
    const char* base_path,
    size_t max_bytes,
    std::vector<kv_ssd_evicted_checkpoint>* evicted = nullptr);

// Stable anonymous conversation hash. If first_user_end is valid, hash the
// complete stable prefix through the first user message; otherwise retain the
// legacy first-1024-token behavior.
uint64_t kv_ssd_hash_conversation(
    const uint32_t* tokens,
    size_t tokens_size,
    size_t first_user_end = 0);

// Scan all conversation directories for a fuzzy prefix match.
// Only considers directories with matching compat_hash (model config).
// Returns conv_hash of best match, or 0 if none found above min_overlap.
uint64_t kv_ssd_find_continuation(
    const char* base_path,
    const uint32_t* tokens, size_t tokens_size,
    float min_overlap,
    uint64_t compat_hash = 0,
    float* out_overlap = nullptr);

// Get maximum turn_id across all conversation directories (for seeding turn counter on restart).
uint32_t kv_ssd_get_max_turn_id_global(const char* base_path);

// Hint that a checkpoint will be needed soon.
// Triggers kernel page cache prefetch (posix_fadvise WILLNEED on Linux,
// readahead on macOS) to overlap SSD I/O with CPU work.
// Safe to call from any thread. No-op if checkpoint is already in RAM.
void kv_ssd_prefetch(kv_ssd_cache* cache, uint64_t checkpoint_id);

// Prefetch all cold checkpoints for a given slot.
// Useful for pre-warming the page cache before a slot is processed.
void kv_ssd_prefetch_slot(kv_ssd_cache* cache, uint32_t slot_id);

#endif // KV_SSD_CACHE_H
