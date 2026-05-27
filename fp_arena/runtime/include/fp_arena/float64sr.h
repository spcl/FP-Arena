// Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// ``fp_arena::float64sr`` -- a double-precision value that uses stochastic
// rounding for every arithmetic result. It stores a plain ``double`` and is a
// trivially-copyable, standard-layout drop-in replacement for ``double``.
//
// Because there is no wider native type, each operation captures its exact
// result as an unevaluated sum (``hi + lo``) using error-free transforms
// (TwoSum for +/-, FMA-based TwoProd for *, FMA residual for /), then rounds
// stochastically (see ``stochastic_rounding.h``). Construction from a smaller
// type (``float``/``int``) is exact; construction from a ``double`` is exact
// (the value is already a double).
#pragma once

#include "stochastic_rounding.h"

namespace fp_arena {

struct float64sr {
    double value;

    float64sr() = default;  // trivial: leaves the type trivially copyable

    FP_ARENA_HD float64sr(double v) : value(v) {}  // exact: v is already a double
    FP_ARENA_HD float64sr(float v) : value(static_cast<double>(v)) {}
    FP_ARENA_HD float64sr(int v) : value(static_cast<double>(v)) {}
    FP_ARENA_HD float64sr(long long v) : value(static_cast<double>(v)) {}

    FP_ARENA_HD operator double() const { return value; }

    FP_ARENA_HD float64sr operator-() const { return from_raw(-value); }  // negation is exact

    static FP_ARENA_HD float64sr from_raw(double v) {
        float64sr r;
        r.value = v;
        return r;
    }

#define FP_ARENA_F64_BINOP(op, fn)                                                                             \
    FP_ARENA_HD float64sr operator op(const float64sr& o) const {                                              \
        return from_raw(detail::fn(value, o.value));                                                           \
    }                                                                                                          \
    FP_ARENA_HD float64sr operator op(double o) const { return from_raw(detail::fn(value, o)); }               \
    FP_ARENA_HD float64sr operator op(float o) const {                                                        \
        return from_raw(detail::fn(value, static_cast<double>(o)));                                            \
    }                                                                                                          \
    FP_ARENA_HD float64sr operator op(int o) const {                                                          \
        return from_raw(detail::fn(value, static_cast<double>(o)));                                            \
    }                                                                                                          \
    friend FP_ARENA_HD float64sr operator op(double l, const float64sr& r) {                                  \
        return from_raw(detail::fn(l, r.value));                                                               \
    }                                                                                                          \
    friend FP_ARENA_HD float64sr operator op(float l, const float64sr& r) {                                   \
        return from_raw(detail::fn(static_cast<double>(l), r.value));                                          \
    }                                                                                                          \
    friend FP_ARENA_HD float64sr operator op(int l, const float64sr& r) {                                     \
        return from_raw(detail::fn(static_cast<double>(l), r.value));                                          \
    }

    FP_ARENA_F64_BINOP(+, add_sr)
    FP_ARENA_F64_BINOP(-, sub_sr)
    FP_ARENA_F64_BINOP(*, mul_sr)
    FP_ARENA_F64_BINOP(/, div_sr)
#undef FP_ARENA_F64_BINOP

#define FP_ARENA_F64_COMPOUND(op, fn)                                                                          \
    FP_ARENA_HD float64sr& operator op(const float64sr& o) {                                                   \
        value = detail::fn(value, o.value);                                                                    \
        return *this;                                                                                          \
    }                                                                                                          \
    FP_ARENA_HD float64sr& operator op(double o) {                                                             \
        value = detail::fn(value, o);                                                                          \
        return *this;                                                                                          \
    }

    FP_ARENA_F64_COMPOUND(+=, add_sr)
    FP_ARENA_F64_COMPOUND(-=, sub_sr)
    FP_ARENA_F64_COMPOUND(*=, mul_sr)
    FP_ARENA_F64_COMPOUND(/=, div_sr)
#undef FP_ARENA_F64_COMPOUND

#define FP_ARENA_F64_CMP(op)                                                                                   \
    FP_ARENA_HD bool operator op(const float64sr& o) const { return value op o.value; }                        \
    FP_ARENA_HD bool operator op(double o) const { return value op o; }                                        \
    FP_ARENA_HD bool operator op(float o) const { return value op static_cast<double>(o); }

    FP_ARENA_F64_CMP(==)
    FP_ARENA_F64_CMP(!=)
    FP_ARENA_F64_CMP(<)
    FP_ARENA_F64_CMP(<=)
    FP_ARENA_F64_CMP(>)
    FP_ARENA_F64_CMP(>=)
#undef FP_ARENA_F64_CMP
};

static_assert(sizeof(float64sr) == sizeof(double), "float64sr must match double layout");

}  // namespace fp_arena
