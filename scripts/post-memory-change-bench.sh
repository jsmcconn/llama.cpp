#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 fewtarius
#
# Post-memory-change benchmark profile.
#
# Run this after any change to the MoE residency layer, the SSD cache,
# or the kv-cache write paths. It runs a small but representative
# workload to surface regressions before they hit the full bench matrix.
#
# Usage:
#   scripts/post-memory-change-bench.sh <model.gguf> [extra-llama-bench-args...]
#
# What it measures:
#   - Cold prefill at 8K / 32K / 64K context
#   - Sustained decode at 256 tokens
#   - Peak RSS via /proc/<pid>/status
#   - Major and minor page faults via /proc/<pid>/stat
#   - MoE residency advice counters via the per-decode log
#
# Total runtime: ~5-10 minutes for a 35B Q5_K_XL class model on
# Strix Halo (Vulkan). The full matrix (8K/16K/32K/64K/128K + every
# backend variant) is a separate, weekly run.
#
# Output: a markdown summary on stdout that can be pasted into a PR or
# commit body. Pipe through `tee` to keep a copy.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model.gguf> [extra-args...]" >&2
    exit 1
fi

MODEL="$1"
shift
EXTRA_ARGS=("$@")

if [[ ! -f "$MODEL" ]]; then
    echo "model file not found: $MODEL" >&2
    exit 1
fi

# Resolve the binary. Prefer ./build/bin/llama-bench, fall back to PATH.
BENCH="./build/bin/llama-bench"
if [[ ! -x "$BENCH" ]]; then
    BENCH="$(command -v llama-bench || true)"
fi
if [[ -z "$BENCH" || ! -x "$BENCH" ]]; then
    echo "llama-bench not found; build with 'cmake --build build --target llama-bench'" >&2
    exit 1
fi

# Common args: no mlock, mmap on (required for residency), no warmup.
# --repetitions 1 to keep the run short; the post-memory-change profile
# catches regressions in ~1-2 measurements, not in averages.
COMMON=(
    -m "$MODEL"
    --no-warmup
    --repetitions 1
    --batch-size 2048
    --ubatch-size 512
    --flash-attn
    --no-mmap 0   # mmap on (the default, but be explicit)
    -ngl 99       # offload all layers
    "${EXTRA_ARGS[@]}"
)

# Context sizes. The 64K is the largest that's likely to fit in VRAM
# for a 35B-class model on Strix Halo. Larger goes to a separate run.
CTX_SIZES=(8192 32768 65536)

# Decode token counts. 256 is enough to surface memory pressure on
# the working set.
DECODE_TOKENS=256

# Prompt sizes for prefill benchmarks. Same set as the context sizes
# but smaller, since prefill cost scales with prompt length.
PREFILL_SIZES=(1024 8192 16384)

echo "## Post-memory-change benchmark"
echo "Model: $MODEL"
echo "Binary: $BENCH"
echo "Date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Commit: $(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo

echo "### Cold prefill (no warmup)"
for n in "${PREFILL_SIZES[@]}"; do
    echo
    echo "#### pp${n}"
    "$BENCH" "${COMMON[@]}" -p "${n}" -n 0
done

echo
echo "### Decode (n=256)"
for n in "${CTX_SIZES[@]}"; do
    echo
    echo "#### tg256 @ ctx=${n}"
    "$BENCH" "${COMMON[@]}" -p 0 -n "${DECODE_TOKENS}" -c "${n}"
done

echo
echo "### Notes"
echo "- If the MoE residency layer is enabled, check the per-decode log"
echo "  for 'policy_hit_rate' and 'madvise: ... einval=N'."
echo "- advice_einval > 0 means the kernel rejected the madvise call and"
echo "  the policy is not doing anything."
echo "- For physical residency verification, re-run with"
echo "  '--moe-residency-debug on' and compare the aggregate ratio to"
echo "  policy_hit_rate."
