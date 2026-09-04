# MoE Expert Residency

Enable running Mixture-of-Experts (MoE) models whose total footprint exceeds available system RAM by keeping only the active expert subset paged in and letting cold expert pages spill to the SSD via `madvise`. On Linux with NVMe, the kernel handles the actual I/O through the page cache; CachyLLama just steers it via `MADV_WILLNEED` (hot path) and `MADV_COLD` (cold path) hints.

## Policy vs. physical residency

CachyLLama controls a *software policy* (which experts should be in the working set). The Linux kernel controls *physical residency* (which pages are actually in RAM). These are not the same thing:

- `madvise()` is advisory. The kernel is free to ignore the hint, evict pages anyway, or keep cold pages resident under low memory pressure.
- An expert is "loaded" in our cache the moment we call `MADV_WILLNEED` on it - but the kernel may have already had those pages resident (so the call is a no-op), or may have decided to evict them despite the hint.
- A 95% "hit rate" reported by the residency layer is the *software policy hit rate* - it tells you the LRU predicted the right experts to keep, not that the kernel kept them.

To actually measure physical residency, use `--moe-residency-debug` (Linux only) which periodically calls `mincore()` on each tracked expert and logs the resident/total page ratio. If that ratio is much lower than the policy hit rate, the kernel is evicting pages we asked it to keep - and the policy is not actually doing anything.

The `llama_moe_residency_stats_get()` API exposes both views:

- `total_hits / total_misses` - software policy hits and misses
- `advice_success / advice_failure / advice_einval` - whether the kernel actually accepted each `madvise()` call
- `invalid_mapping` - calls skipped because the region wasn't page-alignable

`advice_einval > 0` is the most important signal: it means the kernel rejected the advice for the mapping type, and the policy is silently no-oping. If you see this, the advice value being used is invalid for the model's `mmap()` layout.

## Overview

Without residency, a 26 GB MoE model on a 25 GB machine OOMs the kernel even though only ~3% of its weights are touched per token. With residency, the same model loads, runs at ~3 tok/s on CPU-only hardware, and keeps the active subset (~800 MB) resident in physical RAM.

Validated sizes on a 25 GB Flip KB (7840U):

| Model | Size | MoE config | Hit rate (32-cached) |
|---|---|---|---|
| gpt-oss-20b Q6_K_XL | 12 GB | 32 exp / 4 used | 83% |
| Qwen3.6-35B-A3B Q5_K_XL | 26 GB | 256 exp / 8 used | 95.5% |
| Qwen3.6-35B-A3B Q8_K_XL | 39 GB | 256 exp / 8 used | 95.5% |
| gpt-oss-120b Q8_K_XL | 64 GB | 128 exp / 4 used | (loaded) |
| Qwen3-Coder-Next Q8_K_XL | 86 GB | 512 exp / 10 used | (loaded) |

All five models load successfully. The 12 GB gpt-oss runs at normal speed. Larger models are SSD-read-bound at 0.7-3 tok/s on CPU-only hardware; adding `-ngl 99` to offload attention/embedding layers to GPU improves throughput substantially.

## Architecture

The subsystem has three pieces:

1. **Per-token expert selection capture** — `track_expert_activations()` reads the MoE routing tensors (`ffn_moe_argsort-<layer>`, `ffn_moe_topk-<layer>`, or `ffn_moe_probs-<layer>`) from the compute graph after each decode and stores the top-K selected expert IDs per layer in `llama_context::expert_stats`.
2. **Per-layer recency+frequency cache** — a fixed-size pool of expert slots per MoE layer. Each decode's selection touches the relevant slots (marking them hot), promotes them by recency/frequency score, and evicts the lowest-scoring slot via `MADV_COLD` when the layer's resident set overflows.
3. **Cross-session co-activation matrix** — records per-layer and cross-layer expert co-firing counts. Saved to `~/.cachylla/coactivation/{model}.json` on context destruction; reloaded on init. Used to inform future prewarm decisions and (in Phase 2+ planning) predictive prefetch.

### Why `madvise` and not explicit `pread`

NVMe SSDs have very high random read performance (millions of IOPS for small reads, GB/s for sequential). The `madvise` + `MADV_WILLNEED` + `MADV_COLD` path lets the kernel's existing page cache machinery do the I/O, with readahead, write coalescing, and eviction policy that are already well-tuned for NVMe. We don't need explicit `pread()` workers or a custom buffer pool — the kernel does it for us at near-zero overhead.

The trade-off: `madvise` is advisory. The kernel can ignore hints under memory pressure. We measure 99.5%+ *policy hit rate* in steady state on tested models, which is the practical limit of the LRU+R+F prediction; the physical residency rate (measured with `--moe-residency-debug`) tracks the same number on hardware that doesn't have competing memory pressure.

We use `MADV_WILLNEED` on the hot path and `MADV_COLD` on the cold path. Earlier versions used `MADV_DONTNEED` and `MADV_FREE`; both are inappropriate here. `MADV_DONTNEED` is destructive on file-backed mappings (the kernel re-faults from disk on next access, which on Flip-tier hardware turned cold misses into disk page faults and dropped prefill from ~215 t/s to ~56 t/s on a 20 GB MoE model). `MADV_FREE` is rejected with `EINVAL` on the model's `MAP_SHARED | PROT_READ` mapping - the Linux man page is explicit that `MADV_FREE` applies only to *private anonymous* pages. `MADV_COLD` (Linux 5.4+) is the right tool: it is non-destructive (no re-fault needed if the kernel keeps the page) and is valid for file-backed shared mappings.

## CLI flags

```
--moe-expert-residency / --no-moe-expert-residency   master switch (default: disabled)
--moe-resident-per-layer N                         experts kept hot per layer (default: 32)
--moe-prewarm-top-k N                              experts to prewarm at startup (default: 16)
--moe-residency-debug [on|off]                     periodic mincore() sampling (default: off)
--moe-residency-debug-interval N                   decodes between mincore() samples (default: 64)
```

All available as environment variables: `LLAMA_ARG_MOE_EXPERT_RESIDENCY`, `LLAMA_ARG_MOE_RESIDENT_PER_LAYER`, `LLAMA_ARG_MOE_PREWARM_TOP_K`, `LLAMA_ARG_MOE_RESIDENCY_DEBUG`, `LLAMA_ARG_MOE_RESIDENCY_DEBUG_INTERVAL`.

`--moe-residency-debug` is Linux only. It is intended for development and correctness verification, not production: the `mincore()` call costs O(experts) per sample. Tune `--moe-residency-debug-interval` to balance observability against overhead. The output is logged at `INFO` level and includes both per-expert residency ratios and an aggregate ratio across the cache. To verify the residency policy is doing what it claims, run the model with this flag and compare `policy_hit_rate` (from the per-decode summary) against the `aggregate ... ratio` line. The two should track each other within a few percent on hardware without competing memory pressure.

Requires mmap (`--mmap` is the default). Disabling mmap (`--no-mmap`) also disables residency.

## Public API

[include/llama.h:1579-1680](include/llama.h#L1579-L1680) — the user-facing types and functions.

```c
// Per-layer expert activation statistics (cumulative since tracking enabled).
struct llama_expert_stats {
    int32_t n_expert;       // number of experts in this layer
    int32_t n_expert_used;  // number of experts used per token
    uint64_t total_tokens;  // total tokens processed in this layer
    uint64_t * activation_count; // [n_expert] per-expert activation count
};

// Enable/disable expert activation tracking.
LLAMA_API void  llama_expert_tracking_enable(struct llama_context * ctx, bool enable);
LLAMA_API bool  llama_expert_tracking_enabled(const struct llama_context * ctx);

// Read cumulative stats for a specific layer.
LLAMA_API int32_t llama_expert_stats_get(const struct llama_context * ctx,
                                        int32_t layer, struct llama_expert_stats * stats);
LLAMA_API void   llama_expert_stats_reset(struct llama_context * ctx);

// Per-layer snapshot of the most recent decode's top-K expert selection.
// Used by the SSD loader / offload subsystem to pre-load experts that
// will likely fire next (temporal locality).
struct llama_expert_last_selection {
    int32_t n_expert_used;
    int32_t n_tokens;
    const int32_t * selected;   // [n_tokens * n_expert_used], row-major
};

LLAMA_API int32_t llama_expert_last_selected_get(const struct llama_context * ctx,
                                                 int32_t layer,
                                                 struct llama_expert_last_selection * selection);
LLAMA_API void     llama_expert_last_selected_clear(struct llama_context * ctx);

// MoE expert residency config + control.
struct llama_moe_residency_config {
    uint8_t  enabled;
    uint32_t max_resident_per_layer;
    uint8_t  prewarm_on_init;
    uint32_t prewarm_top_k;
    uint8_t  log_per_decode;
};

LLAMA_API struct llama_moe_residency_config llama_moe_residency_config_default(void);
LLAMA_API int32_t llama_moe_residency_enable(struct llama_context * ctx,
                                            const struct llama_moe_residency_config * cfg);
LLAMA_API void     llama_moe_residency_disable(struct llama_context * ctx);

struct llama_moe_residency_stats {
    uint64_t total_hits;       // expert touches that were already loaded
    uint64_t total_misses;     // expert touches that required MADV_WILLNEED
    uint64_t total_evicted;    // experts removed from LRU via MADV_COLD
    uint64_t advice_success;   // madvise() calls accepted by the kernel
    uint64_t advice_failure;   // madvise() calls rejected (any errno)
    uint64_t advice_einval;    // subset of failures: errno == EINVAL
    uint64_t invalid_mapping;  // calls skipped (null/len=0/unalignable)
    bool     uses_madv_cold;   // true on Linux; cold-path advice used
    uint64_t decode_count;     // total decode() calls observed
    uint64_t moe_layer_count;  // number of MoE layers in the model
};

LLAMA_API void llama_moe_residency_stats_get(const struct llama_context * ctx,
                                            struct llama_moe_residency_stats * out);

LLAMA_API void llama_set_model_path(struct llama_context * ctx, const char * path);
```

## Internal data flow

[src/llama-context.cpp:777-867](src/llama-context.cpp#L777-L867) — `track_expert_activations()`. Reads the routing tensors from the compute graph and populates `expert_stats[il].last_selected`. Uses a two-pass read: try `ffn_moe_topk` / `ffn_moe_argsort` (I32), validate the first few entries look like expert IDs; if not, fall back to `ffn_moe_probs` (F32) and compute top-K ourselves.

[src/llama-context.cpp:2049-2097](src/llama-context.cpp#L2049-L2097) — decode hook. After each `track_expert_activations`, calls `llama_moe_residency_touch_layer_selection` for each layer to update the R+F cache with the just-fired experts.

[src/llama-moe-residency.cpp:170-237](src/llama-moe-residency.cpp#L170-L237) — `llama_moe_residency_touch()`. The cache update logic. Score = 0.5 * recency + 0.5 * frequency, evict lowest. Records hits and misses.

[src/llama-moe-coact.cpp:23-77](src/llama-moe-coact.cpp#L23-L77) — co-activation recording. Within-layer pair counts + cross-layer correlations. JSON persistence at `~/.cachylla/coactivation/{model}.json`.

[src/llama-context.cpp:718-770](src/llama-context.cpp#L718-L770) — `sched_reserve()` build hook. Builds the residency state and co-activation matrix after model load, prewarms top-K experts from prior stats if available.

[src/llama-context.cpp:497-501](src/llama-context.cpp#L497-L501) — dtor hook. Saves the co-activation matrix to disk before releasing residency state.

## Why R+F cache and not pure LRU

LRU evicts based on access order alone. R+F (recency + frequency) combines:
- **Recency:** `1 / (1 + current_token - last_access)` — high for recently-used experts
- **Frequency:** `access_count / (1 + current_token - loaded_at)` — high for frequently-used experts

Combined score = 0.5 * recency + 0.5 * frequency. Evict the lowest-scoring slot.

This addresses the FlashMoE finding that pure LRU evicts hot experts 34% of the time when access patterns have both temporal locality and burst patterns.

## Tuning guide

Default values are tuned for Qwen3.6-35B-A3B-class models (256 experts, 8 used). For different model shapes:

| Model class | n_expert | n_used | Recommended `--moe-resident-per-layer` |
|---|---|---|---|
| gpt-oss (32/4) | 32 | 4 | 8-16 |
| Qwen3.6-35B (256/8) | 256 | 8 | 32-64 |
| Qwen3-Coder-Next (512/10) | 512 | 10 | 32-64 |
| Mixtral 8x7B (8/2) | 8 | 2 | 4-8 |

The cache hit rate drops if `--moe-resident-per-layer` is too small to cover the active working set. It doesn't hurt to set it larger than needed — unused slots just sit idle.

`--moe-prewarm-top-k` controls how many experts per layer are pre-paged-in at startup. With persistence (co-activation matrix), we prewarm based on observed usage. Without, we prewarm experts 0..K-1.

## Persistence

The co-activation matrix is saved to `~/.cachylla/coactivation/{model}.json` on context destruction (graceful shutdown via SIGTERM; SIGKILL bypasses this). Reloaded on next context init if available.

The file is plain JSON, ~1-10 MB depending on model size. Schema:

```json
{
  "v": 1,
  "nl": 40,           // number of layers
  "ne": 256,          // number of experts
  "oc": [...],        // observation counts per layer
  "lp": [...],        // layer pair counts [nl][ne*ne]
  "cc": [...]         // cross counts [nl][ne][ne]
}
```

## Limitations

- **Linux only.** Uses `madvise`, `/sys/class/power_supply`, and Linux-specific behavior. macOS has different memory management and won't see the same benefits.
- **Advisory hints.** The kernel can evict our "hot" pages under pressure. In practice, with sensible `--moe-resident-per-layer`, we stay at 95%+ hit rate.
- **Doesn't reduce virtual address space.** The model is still mmap'd in full. Linux overcommit handles this for 64-bit, but on 32-bit systems or with strict overcommit, this won't work.
- **Slow on CPU-only.** Larger models (60+ GB) are SSD-read-bound. Adding GPU offload for non-expert layers (`-ngl 99`) substantially improves throughput.
- **Argsort tensor workaround needed.** Some MoE architectures (notably Qwen3.6 family) reuse the compute graph across ubatches, which can cause `ffn_moe_argsort` / `ffn_moe_topk` tensor storage to hold stale data. We fall back to computing top-K from the F32 `ffn_moe_probs` tensor when this happens. See `track_expert_activations` for the validation logic.

## Testing

Reproducing on a target machine:

```bash
# 1. Build with residency support
cmake --build build

# 2. Run with an MoE model that exceeds RAM
./llama-server \
  -m ./models/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  -ngl 0 --no-warmup \
  --moe-expert-residency \
  --moe-resident-per-layer 32

# 3. Send a request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'

# 4. Look for the residency log line every 16 decodes:
# moe-residency: decodes=144 touches=613 hits=2087 misses=421 evictions=421 hit_rate=83.2%
```

If hit rate is below 80%:
- Increase `--moe-resident-per-layer` (more memory, better hits)
- Verify `--mmap` is enabled (default)
- Check that `--no-warmup` isn't interacting badly (warmup pre-paginates the prompt)

## Future work

- **Predictive prefetch:** Use the co-activation matrix to predict layer N+1's likely experts from layer N's selection. Submit prefetch hint while layer N is computing. Eliminates effective miss latency when predictions are correct.
- **Async SSD prefetch pipeline:** worker thread pool with explicit `pread` for the next-layer experts. More complex than `madvise` and offers diminishing returns on NVMe, but could help on HDDs or low-queue-depth NVMe.
- **Dense FFN streaming:** same approach applied to non-MoE models. Useful for Llama-70B-class on memory-constrained systems. Different code path (no expert routing to predict — every token uses everything).
- **Expert layout rewriter:** GGUF file tool that reorders expert tensors so co-activated experts are contiguous. Doesn't help on NVMe (random reads are cheap), so deprioritized.