// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Regression test for fewtarius/llama-ai issue #6:
//   SSD cache auto-sizing overrode explicit --cache-ssd-hot-ram /
//   --cache-ssd-warm-ram caps at conversation-create time.
//
// The auto-sizer in common/kv-ssd-cache.cpp (kv_ssd_init) used
//   if (c->config.auto_size) { ... override caps ... }
// and the server never cleared auto_size when the user supplied the RAM-cap
// flags, so the configured caps were silently replaced with values derived
// from sysinfo.freeram. On unified-memory hardware (Strix Halo, Apple
// Silicon) this made the cache eat RAM shared with the iGPU.
//
// Fix: in tools/server/server-context.cpp, when either RAM-cap flag is set,
// disable auto-sizing before constructing server_context_page_manager.
//
// These tests verify that:
//   1. When auto_size=false, kv_ssd_init preserves the explicit caps.
//   2. When auto_size=true, kv_ssd_init still overrides (default behavior
//      unchanged for setups that rely on auto-sizing).
//   3. The server-side guard logic collapses to auto_size=false whenever
//      either flag is non-zero (matching the fix).
#undef NDEBUG

#include "kv-ssd-cache.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int tests_run    = 0;
static int tests_failed = 0;

#define RUN(name) do {                              \
    printf("  %-50s ", #name);                      \
    fflush(stdout);                                 \
    tests_run++;                                    \
    try {                                           \
        test_##name();                              \
        printf("OK\n");                             \
    } catch (const std::exception & e) {            \
        printf("FAIL: %s\n", e.what());             \
        tests_failed++;                             \
    }                                               \
} while (0)

// Helper: clean and recreate a scratch directory for the SSD cache.
static fs::path make_scratch(const std::string & name) {
    fs::path p = fs::temp_directory_path() / ("ssd_caps_test_" + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("could not create scratch dir " + p.string()
                                 + ": " + ec.message());
    }
    return p;
}

// Test 1: explicit caps survive when auto_size = false.
// Mirrors the server-side fix: when either --cache-ssd-hot-ram or
// --cache-ssd-warm-ram is set, server-context.cpp now sets cfg.auto_size=false
// before passing it down. The per-conversation cache must honor that.
static void test_explicit_caps_preserved() {
    const size_t HOT_MIB  = 4096;
    const size_t WARM_MIB = 2048;
    const size_t HOT_BYTES  = HOT_MIB  * 1024 * 1024;
    const size_t WARM_BYTES = WARM_MIB * 1024 * 1024;

    kv_ssd_config cfg;
    cfg.hot_ram_bytes  = HOT_BYTES;
    cfg.warm_ram_bytes = WARM_BYTES;
    cfg.auto_size      = false;  // <-- the fix

    fs::path scratch = make_scratch("explicit");
    kv_ssd_cache * c =
        kv_ssd_init(scratch.string().c_str(), &cfg, /*conv_hash=*/0xC0FFEEULL);

    assert(c != nullptr);
    assert(c->config.hot_ram_bytes == HOT_BYTES);
    assert(c->config.warm_ram_bytes == WARM_BYTES);
    assert(c->config.auto_size == false);

    kv_ssd_free(c);
    fs::remove_all(scratch);
}

// Test 2: auto-sizing still kicks in when auto_size = true (default).
// Existing setups that rely on auto-sizing must not regress.
static void test_auto_size_still_runs() {
    // Pick values that would NOT survive auto-sizing on a host with > ~3 GB
    // of free RAM. If auto_size works, the cache overrides these small caps.
    const size_t TINY_HOT  = 16  * 1024 * 1024;  // 16 MiB
    const size_t TINY_WARM = 8   * 1024 * 1024;  // 8 MiB

    kv_ssd_config cfg;
    cfg.hot_ram_bytes  = TINY_HOT;
    cfg.warm_ram_bytes = TINY_WARM;
    cfg.auto_size      = true;   // <-- default; auto-size should kick in
    cfg.memory_reserve = 0.15f;

    fs::path scratch = make_scratch("auto");
    kv_ssd_cache * c =
        kv_ssd_init(scratch.string().c_str(), &cfg, /*conv_hash=*/0xA17ECAFEULL);

    assert(c != nullptr);
    // Auto-sizing replaced the tiny caps with values derived from
    // get_available_ram(). hot + warm should now sum to ~85% of free RAM
    // (memory_reserve=0.15), which on any host with > ~28 MiB free RAM
    // exceeds the 24 MiB combined seed.
    assert(c->config.hot_ram_bytes  > TINY_HOT);
    assert(c->config.warm_ram_bytes > TINY_WARM);

    kv_ssd_free(c);
    fs::remove_all(scratch);
}

// Test 3: explicit caps with auto_size=false do not trigger the auto-sized log
// path. We can't easily intercept LOG_INF from a test binary, so we re-create
// the server-context guard logic and assert it would disable auto-sizing in
// the only configuration where the bug manifested (one or both caps > 0).
static void test_server_guard_logic() {
    // Reproduce the guard from tools/server/server-context.cpp:
    //   cfg.auto_size = (params_base.cache_ssd_hot_ram_mib == 0 &&
    //                    params_base.cache_ssd_warm_ram_mib == 0);
    auto compute = [](int32_t hot, int32_t warm) -> bool {
        return (hot == 0 && warm == 0);
    };

    // Default (both unset) -> auto-size on, preserves legacy behavior.
    assert(compute(0, 0) == true);

    // Either flag set -> auto-size off, caps become hard limits.
    assert(compute(4096, 0)   == false);
    assert(compute(0, 2048)   == false);
    assert(compute(4096, 2048) == false);

    // Non-positive values for either flag also force auto-size off, which
    // is consistent with how the byte caps fall back to defaults in the
    // surrounding code when the flags are <= 0.
    assert(compute(-1, 0) == false);
}

static void test_durable_plan_and_duplicate_suppression() {
    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.hot_ram_bytes = 1024;
    cfg.warm_ram_bytes = 512;
    cfg.durable_min_growth_tokens = 4096;
    cfg.durable_max_age_ms = 600000;

    fs::path scratch = make_scratch("durable-plan");
    kv_ssd_cache * c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xD00DULL);
    assert(c != nullptr);

    std::vector<uint32_t> tokens(8192);
    for (uint32_t i = 0; i < tokens.size(); ++i) tokens[i] = i + 1;
    const std::vector<uint8_t> state(64, 0x5a);

    auto first = kv_ssd_plan_store(c, 0, 0, 4095, 4096, 1,
                                   tokens.data(), tokens.size(), 0, false, false);
    assert(first.action == KV_SSD_STORE_WRITE);
    const uint64_t id = kv_ssd_store(c, 0, state.data(), state.size(), 0, 4095,
                                     4096, 1, tokens.data(), tokens.size());
    assert(id != 0);

    const fs::path checkpoint_path = fs::path(c->model_dir) / ("ckpt-" + std::to_string(id) + ".bin");
    const auto written_at = fs::last_write_time(checkpoint_path);

    auto duplicate = kv_ssd_plan_store(c, 0, 0, 4095, 4096, 2,
                                       tokens.data(), tokens.size(), 0, false, false);
    assert(duplicate.action == KV_SSD_STORE_REUSE);
    assert(duplicate.checkpoint_id == id);
    assert(c->next_id == id + 1);
    assert(fs::last_write_time(checkpoint_path) == written_at);

    auto deferred = kv_ssd_plan_store(c, 7, 0, 4195, 4196, 3,
                                      tokens.data(), tokens.size(), 0, false, false);
    assert(deferred.action == KV_SSD_STORE_SKIP_CADENCE);
    assert(deferred.growth_tokens == 100);
    assert(c->slot_latest.at(7) == id);

    // A same-length branch must be written even when its stored prefix agrees.
    auto branch_tokens = tokens;
    branch_tokens[4095] ^= 0x55;
    auto branch = kv_ssd_plan_store(c, 0, 0, 4095, 4096, 4,
                                    branch_tokens.data(), branch_tokens.size(), 0, false, false);
    assert(branch.action == KV_SSD_STORE_WRITE);

    // Exact prompts with different context positions are not duplicate state.
    auto different_pos = kv_ssd_plan_store(c, 0, 1, 4096, 4096, 4,
                                           tokens.data(), tokens.size(), 0, false, false);
    assert(different_pos.action == KV_SSD_STORE_WRITE);
    // Recurrent checkpoints naturally advance pos_min/pos_max with an append;
    // token-prefix compatibility, rather than equal positions, drives cadence.
    auto advanced_pos = kv_ssd_plan_store(c, 0, 17, 4212, 4196, 4,
                                          tokens.data(), tokens.size(), 0, false, true);
    assert(advanced_pos.action == KV_SSD_STORE_SKIP_CADENCE);
    // Speculative implementation state is optional at append boundaries. A
    // mismatched exact state is not reused, while the self-consistent older
    // target+draft checkpoint remains a valid durable prefix for an append.
    auto different_spec = kv_ssd_plan_store(c, 0, 0, 4095, 4096, 4,
                                            tokens.data(), tokens.size(), 0, false, true);
    assert(different_spec.action == KV_SSD_STORE_WRITE);
    auto different_draft = kv_ssd_plan_store(c, 0, 0, 4195, 4196, 4,
                                             tokens.data(), tokens.size(), 0, true, false);
    assert(different_draft.action == KV_SSD_STORE_WRITE);

    auto growth_due = kv_ssd_plan_store(c, 0, 0, 8191, 8192, 5,
                                        tokens.data(), tokens.size(), 0, false, false);
    assert(growth_due.action == KV_SSD_STORE_WRITE);

    // A small append becomes due when the durable file reaches max age.
    fs::last_write_time(checkpoint_path,
                        fs::file_time_type::clock::now() - std::chrono::seconds(700));
    auto age_due = kv_ssd_plan_store(c, 0, 0, 4195, 4196, 6,
                                     tokens.data(), tokens.size(), 0, false, false);
    assert(age_due.action == KV_SSD_STORE_WRITE);

    kv_ssd_free(c);

    // Exercise the inverse transition as well: a checkpoint containing the
    // optional speculative blob can still govern append cadence when the new
    // boundary has no blob, while an exact state with the blob is reusable.
    kv_ssd_cache * c_spec = kv_ssd_init(scratch.string().c_str(), &cfg, 0xD00EULL);
    assert(c_spec != nullptr);
    const std::vector<uint8_t> spec_state(32, 0x3c);
    const uint64_t spec_id = kv_ssd_store(c_spec, 0, state.data(), state.size(), 0, 4095,
                                          4096, 1, tokens.data(), tokens.size(), 0,
                                          nullptr, 0, spec_state.data(), spec_state.size());
    assert(spec_id != 0);

    auto exact_with_spec = kv_ssd_plan_store(c_spec, 0, 0, 4095, 4096, 2,
                                             tokens.data(), tokens.size(), 0, false, true);
    assert(exact_with_spec.action == KV_SSD_STORE_REUSE);
    assert(exact_with_spec.checkpoint_id == spec_id);

    auto append_without_spec = kv_ssd_plan_store(c_spec, 0, 17, 4212, 4196, 3,
                                                  tokens.data(), tokens.size(), 0, false, false);
    assert(append_without_spec.action == KV_SSD_STORE_SKIP_CADENCE);

    kv_ssd_free(c_spec);
    fs::remove_all(scratch);
}

static void test_oversized_checkpoint_stays_cold() {
    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.hot_ram_bytes = 8;
    cfg.warm_ram_bytes = 4;

    fs::path scratch = make_scratch("oversized");
    kv_ssd_cache * c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xB16B00B5ULL);
    const std::vector<uint8_t> state(64, 0x6b);
    const uint32_t tokens[] = { 1, 2, 3, 4 };
    const uint64_t id = kv_ssd_store(c, 0, state.data(), state.size(), 0, 3,
                                     4, 1, tokens, 4);
    assert(id != 0);
    assert(c->hot_bytes == 0);
    assert(c->index.at(id).tier == KV_TIER_COLD);

    std::vector<uint8_t> loaded;
    assert(kv_ssd_load(c, id, loaded));
    assert(loaded == state);
    assert(c->hot_bytes == 0);
    assert(c->index.at(id).tier == KV_TIER_COLD);

    // Reject a truncated payload before allocating restore vectors from its
    // header sizes.
    const fs::path checkpoint_path = fs::path(c->model_dir) / ("ckpt-" + std::to_string(id) + ".bin");
    fs::resize_file(checkpoint_path, fs::file_size(checkpoint_path) - 1);
    loaded.clear();
    assert(!kv_ssd_load(c, id, loaded));

    kv_ssd_free(c);
    fs::remove_all(scratch);
}

static void test_global_cap_counts_disk_and_preserves_newest() {
    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.hot_ram_bytes = 1;
    cfg.warm_ram_bytes = 1;
    cfg.max_cold_checkpoints = 0;
    fs::path scratch = make_scratch("global-cap");
    const std::vector<uint8_t> state(128, 0x33);
    const uint32_t tokens[] = { 10, 11, 12, 13 };

    auto populate = [&](uint64_t key, const char * prefix) {
        kv_ssd_cache * c = kv_ssd_init(scratch.string().c_str(), &cfg, key, prefix);
        assert(kv_ssd_store(c, 0, state.data(), state.size(), 0, 3, 4, 1,
                            tokens, 4) == 1);
        assert(kv_ssd_store(c, 0, state.data(), state.size(), 0, 3, 4, 2,
                            tokens, 4) == 2);
        kv_ssd_free(c);
    };
    populate(0x11, "");
    populate(0x22, "");
    populate(0x33, "u/");

    const fs::path sample = scratch / "0000000000000011" / "ckpt-1.bin";
    const size_t file_size = (size_t)fs::file_size(sample);
    std::vector<kv_ssd_evicted_checkpoint> evicted;
    const size_t remaining = kv_ssd_enforce_size_cap(
        scratch.string().c_str(), file_size * 4, &evicted);
    assert(remaining <= file_size * 4);
    assert(evicted.size() == 2);
    assert(fs::exists(scratch / "0000000000000011" / "ckpt-2.bin"));
    assert(fs::exists(scratch / "0000000000000022" / "ckpt-2.bin"));
    assert(fs::exists(scratch / "u" / "0000000000000033" / "ckpt-2.bin"));

    evicted.clear();
    const size_t tighter = kv_ssd_enforce_size_cap(
        scratch.string().c_str(), file_size * 2, &evicted);
    assert(tighter <= file_size * 2);
    size_t files_left = 0;
    for (const auto & entry : fs::recursive_directory_iterator(scratch)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.rfind("ckpt-", 0) == 0 &&
            name.size() >= 4 && name.compare(name.size() - 4, 4, ".bin") == 0) files_left++;
    }
    assert(files_left == 2);

    fs::remove_all(scratch);
}

static void test_prefix_retention_and_restart() {
    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.hot_ram_bytes = 1;
    cfg.warm_ram_bytes = 1;
    cfg.max_cold_checkpoints = 4;
    cfg.prefix_checkpoints = 3;
    cfg.durable_min_growth_tokens = 4096;
    const auto scratch = make_scratch("prefix-retention");
    auto * c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xA11, "u/");
    std::vector<uint32_t> tokens(8192, 11);
    const std::vector<uint8_t> state(64, 0x5a);
    auto save = [&](size_t n, uint32_t turn) {
        return kv_ssd_store(c, 0, state.data(), state.size(), 0, n - 1, n,
                            turn, tokens.data(), tokens.size());
    };
    const auto anchor = save(4096, 1);
    kv_ssd_mark_prefix(c, anchor);
    // A distinct useful preamble needs its own exact boundary even though
    // it falls inside the normal durable growth cadence.
    assert(kv_ssd_plan_store(c, 0, 0, 4195, 4196, 2, tokens.data(), tokens.size(),
        0, false, false).action == KV_SSD_STORE_SKIP_CADENCE);
    assert(kv_ssd_plan_store(c, 0, 0, 4195, 4196, 2, tokens.data(), tokens.size(),
        0, false, false, true).action == KV_SSD_STORE_WRITE);
    assert(kv_ssd_plan_store(c, 0, 0, 4095, 4096, 2, tokens.data(), tokens.size(),
        0, false, false, true).action == KV_SSD_STORE_REUSE);
    for (uint32_t turn = 2; turn <= 12; ++turn) save(5000 + turn, turn);
    kv_ssd_on_turn_complete(c, 20);
    assert(c->index.size() == 4);
    assert(c->index.count(anchor) == 1);
    assert(c->hot_bytes == 0 && c->warm_bytes == 0);
    const auto directory = c->model_dir;
    kv_ssd_free(c);
    c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xA11, "u/");
    assert(c->prefix_checkpoint_ids == std::vector<uint64_t>{anchor});
    std::vector<uint8_t> restored;
    assert(kv_ssd_load(c, anchor, restored) && restored == state);
    kv_ssd_free(c);
    // The global cap honors hints from an unloaded namespace but can spill
    // anchors too. No in-memory metadata is available to this operation.
    const size_t file_size = fs::file_size(fs::path(directory) / "ckpt-1.bin");
    assert(kv_ssd_enforce_size_cap(scratch.string().c_str(), file_size * 2) == file_size * 2);
    assert(fs::exists(fs::path(directory) / "ckpt-1.bin"));
    assert(kv_ssd_enforce_size_cap(scratch.string().c_str(), file_size) == file_size);
    assert(!fs::exists(fs::path(directory) / "ckpt-1.bin"));
    c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xA11, "u/");
    assert(c->prefix_checkpoint_ids.empty()); // stale hints are harmless
    kv_ssd_free(c);
    fs::remove_all(scratch);
}

static void test_prefix_retention_is_bounded() {
    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.hot_ram_bytes = 1;
    cfg.warm_ram_bytes = 1;
    cfg.prefix_checkpoints = 3;
    cfg.max_cold_checkpoints = 2;
    const auto scratch = make_scratch("prefix-bound");
    auto * c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xA12);
    const std::vector<uint8_t> state(64, 1);
    const std::vector<uint32_t> tokens(4096, 11);
    for (uint32_t turn = 1; turn <= 4; ++turn) {
        auto id = kv_ssd_store(c, 0, state.data(), state.size(), 0, 4095, 4096,
                              turn, tokens.data(), tokens.size());
        kv_ssd_mark_prefix(c, id);
    }
    assert((c->prefix_checkpoint_ids == std::vector<uint64_t>{2, 3, 4}));
    kv_ssd_mark_prefix(c, 2); // most recently used anchor survives a smaller cap
    kv_ssd_on_turn_complete(c, 20);
    assert(c->index.size() == 2 && c->index.count(2) && c->index.count(4));
    kv_ssd_free(c);
    cfg.prefix_checkpoints = 0;
    c = kv_ssd_init(scratch.string().c_str(), &cfg, 0xA12);
    assert(c->prefix_checkpoint_ids.empty());
    kv_ssd_free(c);
    fs::remove_all(scratch);
}

int main(void) {
    printf("test-ssd-cache-caps: regression suite for issue #6\n");
    printf("====================================================\n\n");

    RUN(explicit_caps_preserved);
    RUN(auto_size_still_runs);
    RUN(server_guard_logic);
    RUN(durable_plan_and_duplicate_suppression);
    RUN(oversized_checkpoint_stays_cold);
    RUN(global_cap_counts_disk_and_preserves_newest);
    RUN(prefix_retention_and_restart);
    RUN(prefix_retention_is_bounded);

    printf("\n====================================================\n");
    printf("Ran %d tests, %d failed\n", tests_run, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
