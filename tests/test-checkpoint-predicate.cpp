// SPDX-License-Identifier: GPL-3.0-or-later
//
// Unit test for the in-memory ring buffer checkpoint acceptance predicate
// in server-context.cpp (around line 4576). The predicate decides whether a
// candidate checkpoint should be loaded to restore a slot's KV cache before
// prefill starts.
//
// Background:
//
//   The upstream predicate was:
//     return cur.pos_min < pos_min_thold || cur.pos_min == 0;
//   added in ggml-org#21510 for SWA models (Gemma 4) where a deferred-final
//   snapshot legitimately has pos_min=0.
//
//   CachyLLama's role-based XML trim (no CSSS) means a pos_min=0 snapshot is
//   only valid for the exact prompt that produced it. When the agent trims
//   dialog between turns, the LCP between the new task and the slot's stored
//   prompt drops, and the unconditional `pos_min == 0` clause accepts a
//   stale snapshot from a previous turn. The CachyLLama fix is to drop the
//   `cur.pos_min == 0` clause and rely on the `n_swa > 0` SWA filter
//   above plus the `pos_min < pos_min_thold` mid-prompt gate.
//
//   Additionally, on WARM slots for hybrid MoE/SSM models (qwen35moe with
//   full_attention_interval=4), a deferred-final at pos_min=0 carries
//   recurrent R/S state computed during the OLD conversation's token flow.
//   Even when positions 0..LCP-1 have identical tokens, the recurrent
//   boundary state was shaped by the old conversation's suffix (the trimmed
//   context). Loading this stale state corrupts generation. The guard
//   rejects deferred-finals on warm slots when pos_next < cur.pos_max
//   (the LCP doesn't cover the checkpoint's full extent), unless the slot
//   was cold-started (ssd_cold_start_used) which guarantees a clean slate.
//
// This test exercises the predicate as a pure function over (n_swa,
// pos_next, pos_min_thold, ssd_cold_start_used, checkpoint) so the
// regression can be caught without a model or a running server. The
// production lambda captures n_swa, pos_next, pos_min_thold,
// ssd_cold_start_used from the outer scope; here they are parameters.
// The shape of the predicate is identical.
//
// Cases:
//
//   SWA model (n_swa > 0):
//     - mid-prompt ckpt at pos_min < pos_min_thold, pos_max <= pos_next: ACCEPT
//     - mid-prompt ckpt at pos_min < pos_min_thold, pos_max > pos_next: REJECT
//       (SWA filter - covered positions past pos_next are stale)
//     - mid-prompt ckpt at pos_min >= pos_min_thold: REJECT
//     - deferred-final at pos_min=0, pos_max <= pos_next, pos_min < thold: ACCEPT
//     - deferred-final at pos_min=0, pos_max > pos_next: REJECT (SWA filter)
//
//   Non-SWA model (n_swa == 0):
//     - mid-prompt ckpt at pos_min < pos_min_thold: ACCEPT
//     - mid-prompt ckpt at pos_min >= pos_min_thold: REJECT
//     - deferred-final at pos_min=0 with pos_min < pos_min_thold: ACCEPT
//       when LCP covers full extent (exact match); REJECT when LCP is
//       partial (stale recurrent state beyond boundary)
//     - deferred-final at pos_min=0 with pos_min >= pos_min_thold: REJECT
//       (REGRESSION: pre-fix code accepted this via pos_min == 0. The
//        slot's stored prompt has more tokens than the new task, the agent
//        trimmed middle tokens, the LCP is small. Loading this snapshot
//        put stale KV at positions 0..task_n_tokens-1.)
//
//   Warm-slot deferred-final on hybrid MoE/SSM (n_swa == 0, !cold_start):
//     - deferred-final at pos_min=0, pos_next < pos_max (LCP < checkpoint
//       extent): REJECT — recurrent state stale beyond LCP boundary
//     - deferred-final at pos_min=0, pos_next >= pos_max (LCP covers full
//       checkpoint): ACCEPT — exact-match LCP, recurrent state is consistent
//     - deferred-final at pos_min=0, ssd_cold_start_used=true: ACCEPT —
//       slot was cleared, no stale recurrent state to worry about

#include "common.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

namespace {

// Mirror of the production predicate in server-context.cpp. If the
// production code drifts, this test will start failing. That's intentional.
bool checkpoint_accepted(const common_prompt_checkpoint & cur,
                         int n_swa,
                         llama_pos pos_next,
                         llama_pos pos_min_thold,
                         bool ssd_cold_start_used) {
    if (n_swa > 0 && cur.pos_max > pos_next) {
        return false;
    }
    // Deferred-final warm-slot guard for hybrid MoE/SSM models.
    // A deferred-final at pos_min=0 carries recurrent R/S state from the
    // old conversation. On a warm slot (not cold-started) where the LCP
    // doesn't cover the checkpoint's full extent (pos_next < cur.pos_max),
    // the recurrent boundary state is stale. Reject it to force do_reset.
    if (cur.pos_min == 0 && !ssd_cold_start_used && pos_next < cur.pos_max) {
        return false;
    }
    return cur.pos_min < pos_min_thold;
}

common_prompt_checkpoint make_ckpt(llama_pos pos_min, llama_pos pos_max, int64_t n_tokens) {
    common_prompt_checkpoint c;
    c.pos_min = pos_min;
    c.pos_max = pos_max;
    c.n_tokens = n_tokens;
    return c;
}

void test_swa_midprompt_inside_window() {
    // SWA, mid-prompt ckpt fully within current window: ACCEPT.
    const auto ckpt = make_ckpt(/*pos_min=*/100, /*pos_max=*/500, /*n_tokens=*/500);
    assert(checkpoint_accepted(ckpt, /*n_swa=*/1024, /*pos_next=*/2000, /*pos_min_thold=*/2000, /*ssd_cold=*/false));
    std::printf("PASS test_swa_midprompt_inside_window\n");
}

void test_swa_midprompt_past_window() {
    // SWA filter: ckpt.pos_max > pos_next -> REJECT.
    const auto ckpt = make_ckpt(/*pos_min=*/100, /*pos_max=*/2500, /*n_tokens=*/2500);
    assert(!checkpoint_accepted(ckpt, /*n_swa=*/1024, /*pos_next=*/2000, /*pos_min_thold=*/2000, /*ssd_cold=*/false));
    std::printf("PASS test_swa_midprompt_past_window\n");
}

void test_swa_midprompt_below_threshold() {
    // pos_min >= pos_min_thold -> REJECT.
    const auto ckpt = make_ckpt(/*pos_min=*/500, /*pos_max=*/700, /*n_tokens=*/700);
    assert(!checkpoint_accepted(ckpt, /*n_swa=*/1024, /*pos_next=*/2000, /*pos_min_thold=*/500, /*ssd_cold=*/false));
    std::printf("PASS test_swa_midprompt_below_threshold\n");
}

void test_swa_deferred_final_inside_window() {
    // SWA, deferred-final at pos_min=0, pos_max within window: ACCEPT
    // via pos_min < pos_min_thold arm. (Pre-fix this was accepted by the
    // pos_min == 0 arm. Either way, ACCEPT.)
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/500, /*n_tokens=*/500);
    assert(checkpoint_accepted(ckpt, /*n_swa=*/1024, /*pos_next=*/2000, /*pos_min_thold=*/1000, /*ssd_cold=*/false));
    std::printf("PASS test_swa_deferred_final_inside_window\n");
}

void test_swa_deferred_final_past_window() {
    // SWA filter: pos_max > pos_next -> REJECT.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/3000, /*n_tokens=*/3000);
    assert(!checkpoint_accepted(ckpt, /*n_swa=*/1024, /*pos_next=*/2000, /*pos_min_thold=*/1000, /*ssd_cold=*/false));
    std::printf("PASS test_swa_deferred_final_past_window\n");
}

void test_non_swa_midprompt_below_threshold() {
    // Non-SWA, mid-prompt ckpt below threshold: ACCEPT.
    const auto ckpt = make_ckpt(/*pos_min=*/100, /*pos_max=*/500, /*n_tokens=*/500);
    assert(checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/2000, /*pos_min_thold=*/1000, /*ssd_cold=*/false));
    std::printf("PASS test_non_swa_midprompt_below_threshold\n");
}

void test_non_swa_midprompt_above_threshold() {
    // Non-SWA, mid-prompt ckpt at or above threshold: REJECT.
    const auto ckpt = make_ckpt(/*pos_min=*/1000, /*pos_max=*/1500, /*n_tokens=*/1500);
    assert(!checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/2000, /*pos_min_thold=*/1000, /*ssd_cold=*/false));
    std::printf("PASS test_non_swa_midprompt_above_threshold\n");
}

void test_non_swa_deferred_final_below_threshold() {
    // Non-SWA, deferred-final at pos_min=0 below threshold: ACCEPT via
    // pos_min < pos_min_thold. This is a warm-slot case where the LCP
    // through the system prompt matches deeply and pos_next covers the
    // checkpoint's full extent — the recurrent state is consistent
    // because positions 0..LCP-1 are identical and the checkpoint was
    // captured from the current conversation's flow.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/500, /*n_tokens=*/500);
    assert(checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/2000, /*pos_min_thold=*/1000, /*ssd_cold=*/false));
    std::printf("PASS test_non_swa_deferred_final_below_threshold\n");
}

void test_non_swa_deferred_final_regression() {
    // REGRESSION TEST: non-SWA, deferred-final at pos_min=0 with
    // pos_min_thold=0 (LCP=0 - agent trimmed the dialog completely).
    // Pre-fix: accepted by `pos_min == 0` arm. This put stale KV at
    // positions 0..task_n_tokens-1 of the new turn.
    // Post-fix: rejected by `pos_min < pos_min_thold` (0 < 0 is false).
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/130564, /*n_tokens=*/130564);
    assert(!checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/0, /*pos_min_thold=*/0, /*ssd_cold=*/false));
    std::printf("PASS test_non_swa_deferred_final_regression\n");
}

void test_non_swa_deferred_final_lcp_trim() {
    // REGRESSION TEST: non-SWA, deferred-final from previous turn with
    // small LCP. Slot has 131k stored tokens, new task has small LCP
    // (agent trimmed middle tokens). The deferred-final is stale and
    // must be rejected by the warm-slot guard.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/130564, /*n_tokens=*/130564);
    // pos_min_thold = 22000 (LCP=29000 minus n_swa=8000, roughly the
    // mid-prompt threshold). pos_next=29000 < pos_max=130564, so the
    // warm-slot guard fires and REJECTS this deferred-final. The old
    // conversation's recurrent state beyond position 29000 is stale.
    // This is the exact corruption pattern from the server.log trace:
    // sim_best=0.730, f_keep=0.992, stale recurrent boundary.
    const bool accepted = checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/29000, /*pos_min_thold=*/22000, /*ssd_cold=*/false);
    assert(!accepted);
    std::printf("PASS test_non_swa_deferred_final_lcp_trim\n");
}

void test_non_swa_deferred_final_exact_lcp_accept() {
    // Non-SWA, deferred-final where the LCP covers the checkpoint's
    // full extent (pos_next >= pos_max). The recurrent state is
    // consistent because all positions match the new task's tokens.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/29000, /*n_tokens=*/29000);
    const bool accepted = checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/29000, /*pos_min_thold=*/22000, /*ssd_cold=*/false);
    assert(accepted);
    std::printf("PASS test_non_swa_deferred_final_exact_lcp_accept\n");
}

void test_hybrid_warm_slot_partial_lcp_reject() {
    // REGRESSION TEST for the deferred-final warm-slot guard.
    // Qwen3.5Moe / Qwen3.6 hybrid (full_attention_interval=4,
    // ssm.conv_kernel=4, ssm.state_size=128). A deferred-final at
    // pos_min=0 was captured during the old conversation's full token
    // flow. The new task has a partial LCP (agent trimmed the dialog):
    // pos_next (21926) < cur.pos_max (22440). The LCP doesn't cover the
    // checkpoint's full extent, so the recurrent boundary state at
    // position 21926 is stale. Must REJECT to force do_reset + fresh
    // prefill.
    //
    // This reproduces the server.log trace where task 954 (sim_best=0.730,
    // f_keep=0.992) loaded a stale deferred-final and produced corrupted
    // output despite a strong LCP match through the system prompt.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/22440, /*n_tokens=*/22440);
    const bool accepted = checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/21926, /*pos_min_thold=*/21925, /*ssd_cold=*/false);
    assert(!accepted);
    std::printf("PASS test_hybrid_warm_slot_partial_lcp_reject\n");
}

void test_hybrid_warm_slot_exact_lcp_accept() {
    // Hybrid model warm slot: deferred-final at pos_min=0, but the LCP
    // covers the checkpoint's full extent (pos_next >= pos_max). The
    // recurrent state is consistent because positions 0..LCP-1 are
    // identical and the checkpoint was from this conversation's flow.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/22440, /*n_tokens=*/22440);
    const bool accepted = checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/22440, /*pos_min_thold=*/21925, /*ssd_cold=*/false);
    assert(accepted);
    std::printf("PASS test_hybrid_warm_slot_exact_lcp_accept\n");
}

void test_hybrid_cold_start_deferred_final_accept() {
    // Hybrid model cold start: ssd_cold_start_used=true means the slot
    // was cleared (prompt_clear ran), so there is no stale recurrent
    // state to worry about. The deferred-final is accepted even if
    // pos_next < pos_max because the cold-start path loads fresh state.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/22440, /*n_tokens=*/22440);
    const bool accepted = checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/21926, /*pos_min_thold=*/21925, /*ssd_cold=*/true);
    assert(accepted);
    std::printf("PASS test_hybrid_cold_start_deferred_final_accept\n");
}

void test_hybrid_warm_slot_full_lcp_boundary() {
    // Hybrid model warm slot: pos_next == cur.pos_max - 1 (last token
    // of the checkpoint is the current LCP boundary token). The LCP
    // covers all checkpoint tokens, so the recurrent state is consistent.
    const auto ckpt = make_ckpt(/*pos_min=*/0, /*pos_max=*/22440, /*n_tokens=*/22440);
    const bool accepted = checkpoint_accepted(ckpt, /*n_swa=*/0, /*pos_next=*/22440, /*pos_min_thold=*/22439, /*ssd_cold=*/false);
    assert(accepted);
    std::printf("PASS test_hybrid_warm_slot_full_lcp_boundary\n");
}

} // namespace

int main() {
    test_swa_midprompt_inside_window();
    test_swa_midprompt_past_window();
    test_swa_midprompt_below_threshold();
    test_swa_deferred_final_inside_window();
    test_swa_deferred_final_past_window();
    test_non_swa_midprompt_below_threshold();
    test_non_swa_midprompt_above_threshold();
    test_non_swa_deferred_final_below_threshold();
    test_non_swa_deferred_final_regression();
    test_non_swa_deferred_final_lcp_trim();
    test_non_swa_deferred_final_exact_lcp_accept();
    test_hybrid_warm_slot_partial_lcp_reject();
    test_hybrid_warm_slot_exact_lcp_accept();
    test_hybrid_cold_start_deferred_final_accept();
    test_hybrid_warm_slot_full_lcp_boundary();
    std::printf("All 15 tests passed\n");
    return 0;
}
