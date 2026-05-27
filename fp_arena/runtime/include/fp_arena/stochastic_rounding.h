// Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// Shared infrastructure for the FP-Arena stochastic-rounding (SR) types.
//
// Stochastic rounding rounds a higher-precision value ``x`` to one of the two
// adjacent representable values, with probability proportional to the distance
// to each. For example, if ``x = 7.2`` and the representable neighbours are
// ``7`` and ``8``, then ``x`` rounds to ``8`` with probability ``0.2`` and to
// ``7`` with probability ``0.8``. Over many operations this removes the
// systematic bias of round-to-nearest, which is the property of interest when
// prototyping reduced-precision pipelines.
//
// Everything here is header-only, ``__host__``/``__device__`` capable, and
// performs no heap allocation, so the SR value types stay trivially copyable
// drop-in replacements for ``float``/``double``.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define FP_ARENA_HD __host__ __device__
#else
#define FP_ARENA_HD
#endif

// The error-free transforms below (TwoSum etc.) rely on strict IEEE-754
// rounding, which is incompatible with ``-ffast-math``. Stochastic rounding
// therefore requires precise math: ``fp_arena.enable_fp_arena_extensions``
// removes ``-ffast-math`` / ``--use_fast_math`` from DaCe's compiler flags, so
// the transforms here are exact and need no ``volatile`` barriers.

namespace fp_arena {
namespace detail {

//==========================================================================
// Bit reinterpretation (defined behaviour, works on host and device).
//==========================================================================
FP_ARENA_HD inline uint64_t f64_to_bits(double x) {
    uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

FP_ARENA_HD inline double bits_to_f64(uint64_t b) {
    double x;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}

FP_ARENA_HD inline double f64_abs(double x) {
    return bits_to_f64(f64_to_bits(x) & 0x7FFFFFFFFFFFFFFFull);
}

FP_ARENA_HD inline double fp_fma(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

//==========================================================================
// Random number generation.
//
// The RNG state lives outside the value types (thread-local on the host,
// derived from the device clock on the GPU), so the SR types themselves stay
// trivially copyable. Quality is "good enough to decorrelate roundings", not
// cryptographic.
//==========================================================================
FP_ARENA_HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

#if !defined(__CUDA_ARCH__)
inline uint64_t& host_rng_state() {
    // Seeded once per thread from the (per-thread-unique) address of the state
    // itself, so different threads decorrelate without any shared mutable state.
    static thread_local uint64_t state = 0;
    static thread_local bool seeded = false;
    if (!seeded) {
        seeded = true;
        state = splitmix64(reinterpret_cast<uintptr_t>(&state) ^ 0xA0761D6478BD642Full);
    }
    return state;
}

// Reseed the calling thread's generator (useful for reproducible tests).
inline void seed(uint64_t value) {
    host_rng_state() = value;
}
#endif

FP_ARENA_HD inline uint64_t next_random_u64() {
#if defined(__CUDA_ARCH__)
    unsigned long long t = clock64();
    unsigned tid = (threadIdx.x + blockIdx.x * blockDim.x) +
                   (threadIdx.y + blockIdx.y * blockDim.y) * 4096u +
                   (threadIdx.z + blockIdx.z * blockDim.z) * 4096u * 4096u;
    return splitmix64(static_cast<uint64_t>(t) ^ (static_cast<uint64_t>(tid) + 1) * 0x9E3779B97F4A7C15ull);
#else
    uint64_t& s = host_rng_state();
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
#endif
}

// Uniform double in [0, 1) from the top 53 random bits.
FP_ARENA_HD inline double random_unit_double() {
    return (next_random_u64() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

//==========================================================================
// fp32 stochastic rounding: round a double to float.
//
// A double keeps 52 mantissa bits, a float 23, so 29 low mantissa bits are
// dropped. Those 29 bits encode the exact position of ``x`` between its two
// neighbouring floats. Adding a uniform 29-bit random value and then
// truncating carries into the kept mantissa with probability equal to that
// fractional position -- i.e. exactly the SR probability. This is pure integer
// arithmetic, so it is immune to ``-ffast-math`` and runs identically on GPU.
//==========================================================================
FP_ARENA_HD inline float stochastic_round_to_float(double x) {
    const uint64_t kExcessMask = 0x000000001FFFFFFFull;  // low 29 mantissa bits
    uint64_t bits = f64_to_bits(x);

    // Already representable as a float (also covers +/-0 and +/-inf): no rounding.
    if ((bits & kExcessMask) == 0)
        return static_cast<float>(x);

    bits += (next_random_u64() & kExcessMask);  // stochastic perturbation
    bits &= ~kExcessMask;                        // truncate to a float-representable double
    return static_cast<float>(bits_to_f64(bits));
}

//==========================================================================
// fp64 stochastic rounding.
//
// There is no wider native type to round from, so the exact result of an
// operation is captured as an unevaluated sum ``hi + lo`` (``hi`` is the
// round-to-nearest result, ``lo`` the exact residual) via error-free
// transforms. We then round ``hi`` toward the neighbour on ``lo``'s side with
// probability ``|lo| / ulp``.
//==========================================================================

// Knuth's TwoSum: hi = fl(a + b), lo = exact rounding error.
FP_ARENA_HD inline void two_sum(double a, double b, double& hi, double& lo) {
    hi = a + b;
    double b_virtual = hi - a;
    double a_virtual = hi - b_virtual;
    double b_err = b - b_virtual;
    double a_err = a - a_virtual;
    lo = a_err + b_err;
}

// TwoProd via FMA: hi = fl(a * b), lo = exact rounding error.
FP_ARENA_HD inline void two_prod(double a, double b, double& hi, double& lo) {
    hi = a * b;
    lo = fp_fma(a, b, -hi);
}

// Next representable double from ``x`` toward +inf (toward_pos) or -inf.
FP_ARENA_HD inline double next_double(double x, bool toward_pos) {
    uint64_t b = f64_to_bits(x);
    bool negative = (b >> 63) != 0;
    if (x == 0.0)
        return toward_pos ? bits_to_f64(1ull) : bits_to_f64((1ull << 63) | 1ull);
    // Moving away from zero increases the magnitude bit pattern; toward zero decreases it.
    if (toward_pos == !negative)
        b += 1;
    else
        b -= 1;
    return bits_to_f64(b);
}

// Stochastically round the unevaluated sum ``hi + lo`` to a double. The
// transform that produced ``hi + lo`` is exact (error-free), and ``ulp`` is a
// power of two, so the probability compare below introduces no rounding either.
FP_ARENA_HD inline double stochastic_round_double(double hi, double lo) {
    if (lo == 0.0)
        return hi;  // exact

    double nxt = next_double(hi, lo > 0.0);
    double ulp = nxt - hi;  // signed gap, same sign as lo; exact (adjacent doubles)
    // Round to ``nxt`` with probability |lo| / |ulp| (in [0, 0.5]).
    if (random_unit_double() * f64_abs(ulp) < f64_abs(lo))
        return nxt;
    return hi;
}

// Convenience wrappers used by the value types.
FP_ARENA_HD inline double add_sr(double a, double b) {
    double hi, lo;
    two_sum(a, b, hi, lo);
    return stochastic_round_double(hi, lo);
}

FP_ARENA_HD inline double sub_sr(double a, double b) {
    double hi, lo;
    two_sum(a, -b, hi, lo);
    return stochastic_round_double(hi, lo);
}

FP_ARENA_HD inline double mul_sr(double a, double b) {
    double hi, lo;
    two_prod(a, b, hi, lo);
    return stochastic_round_double(hi, lo);
}

FP_ARENA_HD inline double div_sr(double a, double b) {
    double q = a / b;
    double r = fp_fma(-q, b, a);  // exact residual a - q*b
    return stochastic_round_double(q, r / b);
}

}  // namespace detail
}  // namespace fp_arena
