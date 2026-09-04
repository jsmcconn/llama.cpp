// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Cross-platform available-RAM query used by auto-sizing code paths in
// common/ and by the Vulkan FA scratch gate in ggml/src/ggml-vulkan/.
//
// Two access patterns:
//   host_available_ram_query()  - returns a real reading or signals unknown;
//                                 callers that must distinguish "real" from
//                                 "fallback guess" should use this.
//   host_available_ram()        - same but with an 8 GiB conservative
//                                 fallback so legacy auto-sizing callers
//                                 don't have to think about it.

#pragma once

#include <cstddef>

namespace common {

// True and sets *out_bytes to currently available RAM (free + reclaimable) on
// supported platforms. False (and *out_bytes = 0) when the answer is unreliable
// (Windows, exotic platforms, kernel older than 3.14, sysinfo() failure).
// Out param must be non-null.
bool host_available_ram_query(std::size_t * out_bytes);

// Currently available RAM in bytes. On Linux/macOS this is the reclaimable
// memory (free + cache / inactive). On unsupported platforms an 8 GiB fallback.
// Safe to call frequently - the underlying syscalls are cheap and read-only.
std::size_t host_available_ram();

}  // namespace common
