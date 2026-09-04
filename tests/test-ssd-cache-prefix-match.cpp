// SPDX-License-Identifier: MIT
#undef NDEBUG

#include "kv-ssd-cache.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <vector>

static kv_ssd_cache * create_cache(const std::filesystem::path & path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path, ec);
    assert(!ec);

    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.no_fsync = true;

    kv_ssd_cache * cache = kv_ssd_init(path.string().c_str(), &cfg, 0x1234ULL);
    assert(cache != nullptr);
    return cache;
}

static void destroy_cache(kv_ssd_cache * cache, const std::filesystem::path & path) {
    kv_ssd_free(cache);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

static bool test_partial_prefix_is_rejected() {
    namespace fs = std::filesystem;

    const fs::path path = fs::temp_directory_path() / "test-ssd-cache-prefix-match-partial";
    kv_ssd_cache * cache = create_cache(path);

    constexpr size_t checkpoint_tokens = 8192;
    std::vector<uint32_t> stored_tokens(checkpoint_tokens);
    for (uint32_t i = 0; i < stored_tokens.size(); ++i) {
        stored_tokens[i] = i;
    }

    std::vector<uint32_t> query_tokens = stored_tokens;
    query_tokens[KV_SSD_TOKEN_PREFIX_MAX] = 9999;
    const std::array<uint8_t, 1> state = { 0 };

    const uint64_t checkpoint_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, checkpoint_tokens - 1, stored_tokens.size(), 0,
        stored_tokens.data(), stored_tokens.size());
    assert(checkpoint_id != 0);

    const uint64_t match = kv_ssd_find_match(
        cache, query_tokens.data(), query_tokens.size(), 0, query_tokens.size());

    destroy_cache(cache, path);
    return match == 0;
}

static bool test_exact_checkpoint_beats_partial_match() {
    namespace fs = std::filesystem;

    const fs::path path = fs::temp_directory_path() / "test-ssd-cache-prefix-match-exact";
    kv_ssd_cache * cache = create_cache(path);

    constexpr size_t checkpoint_tokens = 8192;
    std::vector<uint32_t> query_tokens(checkpoint_tokens);
    for (uint32_t i = 0; i < query_tokens.size(); ++i) {
        query_tokens[i] = i;
    }

    std::vector<uint32_t> partial_tokens = query_tokens;
    partial_tokens[KV_SSD_TOKEN_PREFIX_MAX] = 9999;
    const std::array<uint8_t, 1> state = { 0 };

    const uint64_t exact_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, KV_SSD_TOKEN_PREFIX_MAX - 1,
        KV_SSD_TOKEN_PREFIX_MAX, 1, query_tokens.data(), KV_SSD_TOKEN_PREFIX_MAX);
    assert(exact_id != 0);

    const uint64_t partial_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, checkpoint_tokens - 1, checkpoint_tokens, 2,
        partial_tokens.data(), partial_tokens.size());
    assert(partial_id != 0);

    const uint64_t match = kv_ssd_find_match(
        cache, query_tokens.data(), query_tokens.size(), 0, query_tokens.size());

    destroy_cache(cache, path);
    return match == exact_id;
}

int main() {
    int failures = 0;

    if (!test_partial_prefix_is_rejected()) {
        fprintf(stderr, "partial prefix selected a checkpoint\n");
        failures++;
    }

    if (!test_exact_checkpoint_beats_partial_match()) {
        fprintf(stderr, "partial checkpoint beat an exact checkpoint\n");
        failures++;
    }

    return failures;
}
