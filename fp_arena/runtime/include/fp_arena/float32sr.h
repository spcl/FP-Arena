// Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// ``fp_arena::float32sr`` -- a single-precision value that uses stochastic
// rounding for every arithmetic result. It stores a plain ``float`` and is
// therefore a trivially-copyable, standard-layout drop-in replacement for
// ``float`` (same size, same ABI, usable on the GPU).
//
// Each binary operation is evaluated in ``double`` (exact for +, -, * on
// float operands; faithfully rounded for /), then stochastically rounded back
// to ``float``. Construction from a ``float`` is exact; construction from a
// ``double`` rounds stochastically.
#pragma once

#include "stochastic_rounding.h"

namespace fp_arena {

struct float32sr {
    float value;

    float32sr() = default;  // trivial: leaves the type trivially copyable

    FP_ARENA_HD float32sr(float v) : value(v) {}  // exact
    FP_ARENA_HD float32sr(double v) : value(detail::stochastic_round_to_float(v)) {}
    FP_ARENA_HD float32sr(int v) : value(detail::stochastic_round_to_float(static_cast<double>(v))) {}
    FP_ARENA_HD float32sr(long long v) : value(detail::stochastic_round_to_float(static_cast<double>(v))) {}

    FP_ARENA_HD operator float() const { return value; }

    FP_ARENA_HD float32sr operator-() const { return float32sr(-value); }  // negation is exact

// op between two SR values, and mixed with float/double/int, always rounding
// the (double-precision) result stochastically back to float.
#define FP_ARENA_F32_BINOP(op)                                                                                  \
    FP_ARENA_HD float32sr operator op(const float32sr& o) const {                                              \
        return float32sr(static_cast<double>(value) op static_cast<double>(o.value));                          \
    }                                                                                                          \
    FP_ARENA_HD float32sr operator op(double o) const { return float32sr(static_cast<double>(value) op o); }   \
    FP_ARENA_HD float32sr operator op(float o) const {                                                        \
        return float32sr(static_cast<double>(value) op static_cast<double>(o));                                \
    }                                                                                                          \
    FP_ARENA_HD float32sr operator op(int o) const {                                                          \
        return float32sr(static_cast<double>(value) op static_cast<double>(o));                                \
    }                                                                                                          \
    friend FP_ARENA_HD float32sr operator op(double l, const float32sr& r) {                                  \
        return float32sr(l op static_cast<double>(r.value));                                                   \
    }                                                                                                          \
    friend FP_ARENA_HD float32sr operator op(float l, const float32sr& r) {                                   \
        return float32sr(static_cast<double>(l) op static_cast<double>(r.value));                              \
    }                                                                                                          \
    friend FP_ARENA_HD float32sr operator op(int l, const float32sr& r) {                                     \
        return float32sr(static_cast<double>(l) op static_cast<double>(r.value));                              \
    }

    FP_ARENA_F32_BINOP(+)
    FP_ARENA_F32_BINOP(-)
    FP_ARENA_F32_BINOP(*)
    FP_ARENA_F32_BINOP(/)
#undef FP_ARENA_F32_BINOP

#define FP_ARENA_F32_COMPOUND(op, base)                                                                        \
    FP_ARENA_HD float32sr& operator op(const float32sr& o) {                                                   \
        value = detail::stochastic_round_to_float(static_cast<double>(value) base static_cast<double>(o.value)); \
        return *this;                                                                                          \
    }                                                                                                          \
    FP_ARENA_HD float32sr& operator op(double o) {                                                             \
        value = detail::stochastic_round_to_float(static_cast<double>(value) base o);                          \
        return *this;                                                                                          \
    }

    FP_ARENA_F32_COMPOUND(+=, +)
    FP_ARENA_F32_COMPOUND(-=, -)
    FP_ARENA_F32_COMPOUND(*=, *)
    FP_ARENA_F32_COMPOUND(/=, /)
#undef FP_ARENA_F32_COMPOUND

#define FP_ARENA_F32_CMP(op)                                                                                   \
    FP_ARENA_HD bool operator op(const float32sr& o) const { return value op o.value; }                        \
    FP_ARENA_HD bool operator op(float o) const { return value op o; }                                         \
    FP_ARENA_HD bool operator op(double o) const { return static_cast<double>(value) op o; }

    FP_ARENA_F32_CMP(==)
    FP_ARENA_F32_CMP(!=)
    FP_ARENA_F32_CMP(<)
    FP_ARENA_F32_CMP(<=)
    FP_ARENA_F32_CMP(>)
    FP_ARENA_F32_CMP(>=)
#undef FP_ARENA_F32_CMP
};

static_assert(sizeof(float32sr) == sizeof(float), "float32sr must match float layout");

}  // namespace fp_arena
