// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
#undef NDEBUG

// Regression test for kv_ssd_find_continuation long-checkpoint invisibility.
//
// The continuation scan scored each candidate by matches/rec.n_tokens,
// where matches is bounded by KV_SSD_TOKEN_PREFIX_MAX (4096). With the
// default 0.90 min_overlap, any checkpoint with n_tokens > ~4551 could
// never reach threshold even on perfect prefix alignment
// (4096/8192 = 0.5). Long-running conversations went cold without
// their continuation opportunity being detected on server restart.
//
// Fix: normalize by rec.token_count (the verifiable prefix) instead of
// rec.n_tokens (the full state extent). token_count is bounded by
// KV_SSD_TOKEN_PREFIX_MAX, so the ratio tops out at 1.0 for any
// checkpoint.

#include "kv-ssd-cache.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <vector>

static kv_ssd_cache * create_cache(const std::filesystem::path & path, uint64_t conv_hash) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path, ec);
    assert(!ec);

    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.no_fsync = true;

    kv_ssd_cache * cache = kv_ssd_init(path.string().c_str(), &cfg, conv_hash);
    assert(cache != nullptr);
    return cache;
}

static void destroy_cache(kv_ssd_cache * cache, const std::filesystem::path & path) {
    kv_ssd_free(cache);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// Continuation matching must find a long checkpoint on perfect prefix
// alignment regardless of how far past KV_SSD_TOKEN_PREFIX_MAX its
// n_tokens extends.
static bool test_continuation_finds_long_checkpoint() {
    namespace fs = std::filesystem;

    const fs::path base = fs::temp_directory_path() / "test-ssd-cache-continuation-long";
    const uint64_t conv_hash = 0xABCDEFULL;

    kv_ssd_cache * cache = create_cache(base, conv_hash);

    constexpr size_t  PREFIX   = KV_SSD_TOKEN_PREFIX_MAX;  // 4096
    constexpr uint64_t N_TOKENS = PREFIX * 2;               // 8192

    std::vector<uint32_t> tokens(PREFIX);
    for (uint32_t i = 0; i < PREFIX; ++i) {
        tokens[i] = i;
    }
    const std::array<uint8_t, 1> state = { 0 };

    const uint64_t ckpt_id = kv_ssd_store(
        cache, 0, state.data(), state.size(),
        0, (int32_t) N_TOKENS - 1, N_TOKENS, 1,
        tokens.data(), tokens.size());
    assert(ckpt_id != 0);

    // Free writes the index file so kv_ssd_find_continuation (which walks
    // the base directory) can discover the conversation on its own.
    kv_ssd_free(cache);

    const uint64_t found = kv_ssd_find_continuation(
        base.string().c_str(),
        tokens.data(), tokens.size(),
        0.90f, /*compat_hash=*/0, /*out_overlap=*/nullptr);

    std::error_code ec;
    std::filesystem::remove_all(base, ec);
    return found == conv_hash;
}

int main() {
    int failures = 0;

    if (!test_continuation_finds_long_checkpoint()) {
        fprintf(stderr, "continuation matching missed a long checkpoint\n");
        failures++;
    }

    return failures;
}
