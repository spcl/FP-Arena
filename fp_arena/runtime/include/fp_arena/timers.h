// Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// Header-only phase timers: the runner brackets each SDFG phase with start/stop
// tasklets, read back over the C ABI (fp_arena_timer_*).
//
#pragma once

#include <chrono>
#include <cstddef>
#include <cstring>
#include <vector>

#if defined(FP_ARENA_TIMER_GPU)
#include <cuda_runtime.h>
#endif


namespace fp_arena {
namespace timer {
namespace {

std::vector<double> g_storage;  // [invocation * nslots + slot] -> milliseconds
std::vector<double> g_t0;       // per-slot open-interval start
int g_nslots = 1;
long g_inv = -1;                // -1 so the first begin_invocation lands on row 0

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

void device_sync() {
#if defined(FP_ARENA_TIMER_GPU)
    cudaDeviceSynchronize();
#endif
}

void begin_invocation(int nslots) {
    g_nslots = nslots;
    g_t0.resize(nslots);
    g_inv += 1;
    g_storage.resize(static_cast<std::size_t>(g_inv + 1) * nslots, 0.0);
}

void start(int slot) {
    device_sync();
    g_t0[slot] = now_ms();
}

void stop(int slot) {
    device_sync();
    g_storage[static_cast<std::size_t>(g_inv) * g_nslots + slot] += now_ms() - g_t0[slot];
}

}  // namespace
}  // namespace timer
}  // namespace fp_arena

// extern "C": unmangled names, so ctypes/dlsym finds the readback by string.
extern "C" {

long fp_arena_timer_count() {
    return static_cast<long>(fp_arena::timer::g_storage.size());
}

void fp_arena_timer_copy(double* out) {
    const auto& s = fp_arena::timer::g_storage;
    if (!s.empty()) std::memcpy(out, s.data(), s.size() * sizeof(double));
}

void fp_arena_timer_reset() {
    fp_arena::timer::g_storage.clear();
    fp_arena::timer::g_inv = -1;
}

}  // extern "C"
