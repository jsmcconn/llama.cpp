// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius

#include "host-ram.h"

#include <cstdio>
#include <cstring>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#endif

namespace common {

#ifdef __linux__
// Read /proc/meminfo MemAvailable. This is the kernel-calculated estimate of
// memory available for new allocations without swapping - it includes the
// reclaimable page cache. Available since Linux 3.14 (2014).
// Returns 0 if the file is unreadable or MemAvailable line is missing.
static std::size_t read_meminfo_available() {
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    std::size_t result = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            unsigned long long kb = 0;
            if (std::sscanf(line + 13, " %llu kB", &kb) == 1) {
                result = static_cast<std::size_t>(kb) * 1024ULL;
            }
            break;
        }
    }
    std::fclose(f);
    return result;
}
#endif

static bool host_available_ram_impl(std::size_t * out_bytes) {
    if (!out_bytes) return false;
#ifdef __linux__
    std::size_t avail = read_meminfo_available();
    if (avail == 0) {
        // Kernel older than 3.14 - fall back to sysinfo.freeram (genuinely
        // free RAM only, no reclaimable cache). Conservative but real.
        struct sysinfo info;
        if (sysinfo(&info) != 0) {
            *out_bytes = 0;
            return false;
        }
        avail = static_cast<std::size_t>(info.freeram) * info.mem_unit;
    }
    *out_bytes = avail;
    return true;
#elif defined(__APPLE__)
    // free + inactive: free_count is genuinely unallocated, inactive_count is
    // pages the page cache can evict on demand (Apple's "reclaimable" bucket).
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS || page_size == 0) {
        *out_bytes = 0;
        return false;
    }
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        host, HOST_VM_INFO64, (host_info64_t) &vm_stats, &count);
    if (kr != KERN_SUCCESS) {
        *out_bytes = 0;
        return false;
    }
    std::size_t free_pages = static_cast<std::size_t>(vm_stats.free_count) +
                             static_cast<std::size_t>(vm_stats.inactive_count);
    *out_bytes = free_pages * static_cast<std::size_t>(page_size);
    return true;
#else
    *out_bytes = 0;
    return false;
#endif
}

bool host_available_ram_query(std::size_t * out_bytes) {
    return host_available_ram_impl(out_bytes);
}

std::size_t host_available_ram() {
    std::size_t bytes = 0;
    if (host_available_ram_impl(&bytes)) {
        return bytes;
    }
    // Legacy callers (SSD cache auto-sizing) want a number to plan against even
    // when the platform can't answer reliably. 8 GiB is conservative and
    // matches the original hardcoded fallback these callers used to have.
    return 8ULL * 1024 * 1024 * 1024;
}

}  // namespace common
