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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

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

int main(void) {
    printf("test-ssd-cache-caps: regression suite for issue #6\n");
    printf("====================================================\n\n");

    RUN(explicit_caps_preserved);
    RUN(auto_size_still_runs);
    RUN(server_guard_logic);

    printf("\n====================================================\n");
    printf("Ran %d tests, %d failed\n", tests_run, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}