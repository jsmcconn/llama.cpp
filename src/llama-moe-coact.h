// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius
//
// Co-activation matrix for MoE expert selection prediction.
//
// Tracks which experts tend to fire together within a layer and across
// adjacent layers. After enough observation, the matrix can predict which
// experts are likely to fire next given the currently-selected experts.
//
// Persistence: the matrix is saved to ~/.cachylla/coactivation/{model}.json
// and reloaded on context init if available. This makes predictions useful
// even on the first token of a session for previously-seen models.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

namespace llama_moe_coact {

struct matrix {
    // Per-layer: pair counts of (expert_a, expert_b) firing together.
    // layer_pair_counts[layer][expert_a * num_experts + expert_b] = count.
    // Symmetric: counts[a][b] == counts[b][a] (counts[a][a] is self-count).
    std::vector<std::vector<uint32_t>> layer_pair_counts;
    // Cross-layer: counts[layer][expert_at_N][expert_at_N+1].
    std::vector<std::vector<std::vector<uint32_t>>> cross_counts;
    // Total observations per layer (decodes that produced a selection).
    std::vector<uint32_t> observation_counts;

    int num_layers = 0;
    int num_experts = 0;

    bool has_data() const {
        for (auto c : observation_counts) {
            if (c >= 10) return true;
        }
        return false;
    }
};

// Initialize an empty matrix sized for the given model.
void init(matrix & m, const struct llama_model & model);

// Record which experts fired at a layer for one token. Updates both
// within-layer pair counts and cross-layer correlation with the previous
// layer's selection.
//
// `selected_experts` must have exactly model.hparams.n_expert_used entries.
void record(matrix & m,
            int layer,
            const int32_t * selected_experts,
            int n_selected);

// Record cross-layer correlation between experts at layer N and experts at
// layer N+1. Used by the caller to pair current and previous decode's
// per-token selections.
void record_cross_layer(matrix & m,
                        int layer,
                        const int32_t * selected_n,
                        int n_n,
                        const int32_t * selected_n1,
                        int n_n1);

// Predict which experts are likely to co-fire with `observed` at the same
// layer (top-k), excluding the observed experts themselves. Returns the
// expert IDs in descending order of predicted probability.
//
// Returns up to top_k expert IDs; may return fewer if no data is available
// or only a few non-zero scores exist.
std::vector<int32_t> predict_same_layer(
        const matrix & m,
        int layer,
        const int32_t * observed,
        int n_observed,
        int top_k);

// Predict which experts will fire at layer N+1 given observed selections
// at layer N. top_k limits the number of returned IDs.
std::vector<int32_t> predict_next_layer(
        const matrix & m,
        int layer,
        const int32_t * observed,
        int n_observed,
        int top_k);

// Path to the persistence file for a given model path.
std::string persistence_path(const std::string & model_path);

// Save matrix to disk as JSON. Returns true on success.
bool save(const matrix & m, const std::string & path);

// Load matrix from disk. Returns true on success; `m` is left unmodified
// on failure.
bool load(matrix & m, const std::string & path);

} // namespace llama_moe_coact