// Regression test for issue #8: "Invalid input batch during second turn in OpenCode"
//
// Sequence that triggers the bug:
//   1. Decode a prompt of N tokens (positions [0, N)).
//   2. Continue decoding R additional tokens to simulate "generation beyond the
//      checkpoint boundary" (positions [N, N+R)). The SSD/in-memory checkpoint
//      ends at position N, but the live model state at save time extends to
//      N+R because the previous turn's response tokens are still in the cache.
//   3. Save the state with flags=0 (FULL state - this is what server_ssd_cache
//      store does on disk).
//   4. Load the state into a fresh context with the same seq_id.
//      After load: both mem_attn and mem_recr hold cells for positions
//      [0, N+R) on that seq_id.
//   5. Call llama_memory_seq_rm_attn_only([N, -1)) - simulating what
//      server-context.cpp:4188 does after a checkpoint restore to truncate
//      cells beyond the checkpoint end.
//
// Expected (with the fix): seq_pos_max == N - 1 (the position of the last
//                          checkpoint token). llama_batch_init validation
//                          passes for a new batch starting at position N
//                          (X + 1 == Y).
//
// Bug (without the fix):    seq_pos_max == N+R - 1 - mem_recr still reports
//                          the stale generation positions, hybrid seq_pos_max
//                          (min of attn, recr) returns min(N-1, N+R-1) ==
//                          N-1 actually in theory but in practice mem_attn's
//                          seq_pos_max returned after seq_rm can stay at the
//                          pre-rollback value because the cells weren't
//                          reached by seq_rm's iteration order, leaving
//                          seq_pos_max = N+R-1.
//
// The full seq_rm path (test-recurrent-state-rollback) cannot be used here
// because it triggers the n_rs_seq rollback-exceeded failure for R > n_rs_seq.
// seq_rm_attn_only is the only way to clean up after a checkpoint restore when
// the saved positions extend past the restored checkpoint's logical end.
//
// The same shape of bug also affects DeepSeek-V4 (DSV4) models, which use
// llama_kv_cache_dsv4 (not the hybrid mem_attn + mem_recr layout). DSV4's
// seq_pos_min intentionally returns kv_raw->seq_pos_max so server-context
// cannot roll back via checkpoint search, so a "p0 < seq_pos_min" safety
// gate is broken-by-design on DSV4 (cur_min == seq_pos_max rejects every
// in-range p0). The fix on the DSV4 side removes that gate from
// llama_kv_cache_dsv4::seq_rm so the truncation path actually runs and
// drops stale kv_raw cells past the LCP, letting llama_batch_init's
// "Y = X + 1" check pass.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <vector>

static bool model_is_dsv4(const llama_model * model) {
    char arch[64] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        return false;
    }
    return strcmp(arch, "deepseek4") == 0;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens,
                          uint32_t count, llama_pos pos_start) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        common_batch_add(batch, tokens[i], pos_start + (llama_pos) i, { 0 }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    // The bug is hybrid-specific (mem_recr stale positions) or DSV4-specific
    // (kv_raw cells beyond the checkpoint end survive the seq_rm safety gate).
    // For all other architectures (dense attention, plain recurrent), the
    // seq_pos tracking already lines up with the cells and the test is not
    // informative.
    const bool is_dsv4 = model_is_dsv4(model);
    if (!llama_model_is_hybrid(model) && !is_dsv4) {
        fprintf(stderr, "%s : skipping for non-hybrid, non-DSV4 model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);

    const uint32_t n_rs_seq = 8;
    llama_context * ctx_src = make_ctx(params, model, n_rs_seq);
    llama_context * ctx_dst = make_ctx(params, model, n_rs_seq);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }
    if (!is_dsv4 && llama_n_rs_seq(ctx_src) == 0) {
        // Hybrid models need n_rs_seq > 0 to reproduce the bug via mem_recr.
        // DSV4 doesn't use n_rs_seq, so this check is meaningless there.
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        // no-op vocab: synthesize distinct tokens
        for (int i = 1; i <= 32; ++i) tokens.push_back(i);
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog many times many times over and over and over again", true);
    }
    // Need enough tokens for N + R where R > n_rs_seq.
    const uint32_t N = 4;                       // checkpoint logical end (positions [0, N))
    const uint32_t R = n_rs_seq + 4;             // generation beyond checkpoint (positions [N, N+R))
    const uint32_t total = N + R;
    if (tokens.size() < total + 1) {
        fprintf(stderr, "%s : not enough tokens (need %u, got %zu)\n", __func__, total + 1, tokens.size());
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 1;
    }
    tokens.resize(total + 1);

    // Step 1: decode the prompt portion [0, N) on the source.
    if (!decode_tokens(ctx_src, tokens, N, 0)) {
        fprintf(stderr, "%s : failed to decode prompt [0, N=%u)\n", __func__, N);
        return 1;
    }

    // Step 2: decode generation tokens [N, N+R) - simulating the previous
    // turn's response tokens that extended the cache beyond the checkpoint.
    {
        std::vector<llama_token> gen_tokens(tokens.begin() + N, tokens.begin() + N + R);
        if (!decode_tokens(ctx_src, gen_tokens, R, N)) {
        fprintf(stderr, "%s : failed to decode generation [N=%u, N+R=%u)\n", __func__, N, N + R);
        return 1;
        }
    }

    // Step 3: save the state with flags=0 (FULL state, attn+recr - what
    // server_ssd_cache::store does).
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);

    const llama_pos pre_pos_max =
        llama_memory_seq_pos_max(llama_get_memory(ctx_src), 0);
    if (pre_pos_max < (llama_pos) (N + R - 1)) {
        fprintf(stderr, "%s : precondition failed: seq_pos_max (%d) < N+R-1 (%u)\n",
                __func__, (int) pre_pos_max, N + R - 1);
        return 1;
    }

    // Step 4: load into the fresh context - simulates SSD checkpoint restore.
    ckpt.load_tgt(ctx_dst, 0, 0);

    const llama_pos loaded_pos_max =
        llama_memory_seq_pos_max(llama_get_memory(ctx_dst), 0);
    if (loaded_pos_max != pre_pos_max) {
        fprintf(stderr, "%s : load failed: loaded pos_max=%d != src pos_max=%d\n",
                __func__, (int) loaded_pos_max, (int) pre_pos_max);
        return 1;
    }

    // Step 5: this is the call from server-context.cpp:4188. It is supposed
    // to truncate stale positions past the checkpoint end. Bug: on hybrid
    // models it only clears mem_attn, leaving mem_recr's stale positions
    // intact so seq_pos_max keeps reporting N+R-1 instead of N-1.
    llama_memory_seq_rm_attn_only(llama_get_memory(ctx_dst), 0, N, -1);

    const llama_pos post_pos_max =
        llama_memory_seq_pos_max(llama_get_memory(ctx_dst), 0);

    // Validation success criterion. llama_batch_init checks:
    //   Y (new batch min pos) == X (memory seq_pos_max) + 1
    // For a batch starting at position N, we need X == N - 1.
    // Any value < N satisfies the validation (-1 is treated as "no constraint"
    // by llama_batch_init; N-1 is the tight expected value).
    //
    // Before the fix, post_pos_max == N + R - 1 which makes Y (=N) fail the
    // "Y = X + 1" check and llama_decode returns "Invalid input batch".
    if (post_pos_max >= (llama_pos) N) {
        fprintf(stderr, "%s : FAIL - seq_pos_max = %d, expected < %d (issue #8 regression)\n",
                __func__, (int) post_pos_max, (int) N);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 1;
    }

    // Step 6: actually attempt a llama_decode at position N, which is what
    // server-context.cpp hits via the validation in llama_batch_init. On the
    // unfixed code path this returns "Invalid input batch" (issue #8). On
    // the fixed path it succeeds because both mem_attn and mem_recr report
    // positions strictly below N.
    {
        std::vector<llama_token> extra(tokens.begin() + N, tokens.end());
        const size_t extra_count = extra.size();
        llama_batch batch = llama_batch_init(extra.size(), 0, 1);
        for (size_t i = 0; i < extra_count; ++i) {
            common_batch_add(batch, extra[i], N + (llama_pos) i, { 0 },
                             i + 1 == extra_count);
        }
        const int decode_rc = llama_decode(ctx_dst, batch);
        llama_batch_free(batch);
        if (decode_rc != 0) {
            fprintf(stderr, "%s : FAIL - llama_decode returned %d after seq_rm_attn_only "
                    "(issue #8: \"Invalid input batch\")\n",
                    __func__, decode_rc);
            llama_free(ctx_src);
            llama_free(ctx_dst);
            return 1;
        }
    }

    fprintf(stdout, "%s : OK - %s, seq_pos_max = %d (truncated from %d by seq_rm_attn_only)\n",
            __func__, is_dsv4 ? "DSV4" : "hybrid", (int) post_pos_max, (int) loaded_pos_max);

    llama_free(ctx_src);
    llama_free(ctx_dst);
    return 0;
}
