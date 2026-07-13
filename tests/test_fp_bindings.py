"""Tests for the nanobind fp<Exp, Prec> value classes (fp_arena.native)."""

import math
from fractions import Fraction

import numpy as np

import fp_arena

core = fp_arena.native


# -- exact reference: round a rational to (exp_bits, prec) with RNE ----------


def _round_fp(x: Fraction, exp_bits: int, prec: int):
    """Round-to-nearest-even of an exact rational into the fp format.

    :returns: the rounded value as a Fraction, or +-inf on overflow.
    """
    if x == 0:
        return Fraction(0)
    sign = 1 if x > 0 else -1
    ax = abs(x)
    bias = 2 ** (exp_bits - 1) - 1
    qmin = (1 - bias) - (prec - 1)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if ax < Fraction(2) ** e:
        e -= 1
    q = max(e - (prec - 1), qmin)
    scaled = ax / Fraction(2) ** q
    m = int(scaled)
    frac = scaled - m
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and m % 2 == 1):
        m += 1
    if m == 0:
        return Fraction(0)
    if m.bit_length() - 1 + q > bias:
        return sign * math.inf
    return sign * m * Fraction(2) ** q


def test_fraction_reference_fp23_46():
    """+,-,*,/ on fp23_46 match exact rational arithmetic rounded with RNE."""
    f = core.fp23_46
    rng = np.random.default_rng(3)
    for i in range(400):
        scale = 10.0 ** rng.uniform(-30, 30)
        a = rng.uniform(-1, 1) * scale
        # close magnitudes half the time to exercise cancellation
        b = (
            a * (1 + rng.uniform(-1e-13, 1e-13))
            if i % 2
            else rng.uniform(-1, 1) * scale
        )
        fa, fb = f(a), f(b)
        va, vb = Fraction(float(fa)), Fraction(float(fb))  # exact inputs
        assert (fa < fb) == (va < vb) and (fa == fb) == (va == vb)
        assert (fa >= fb) == (va >= vb) and (fa > fb) == (va > vb)
        if vb == 0:
            continue
        for got, expect in [
            (fa + fb, _round_fp(va + vb, 23, 46)),
            (fa - fb, _round_fp(va - vb, 23, 46)),
            (fa * fb, _round_fp(va * vb, 23, 46)),
            (fa / fb, _round_fp(va / vb, 23, 46)),
        ]:
            assert Fraction(float(got)) == expect, (a, b, float(got), expect)


def test_float32_bit_exact():
    """fp8_24 arithmetic is bit-identical to numpy float32."""
    f = core.fp8_24
    rng = np.random.default_rng(4)
    for _ in range(2000):
        a, b = (
            np.float32(rng.uniform(-1, 1) * 10.0 ** rng.uniform(-44, 38))
            for _ in range(2)
        )
        fa, fb = f(float(a)), f(float(b))
        with np.errstate(over="ignore"):  # overflow to inf is part of the test
            cases = [
                (fa + fb, a + b),
                (fa - fb, a - b),
                (fa * fb, a * b),
                (fa / fb, a / b),
            ]
        for got, ref in cases:
            assert np.float32(float(got)).tobytes() == ref.tobytes(), (a, b)
        assert (fa < fb) == (a < b) and (fa == fb) == (a == b)


def test_float64_bit_exact():
    f = core.fp11_53
    rng = np.random.default_rng(5)
    for _ in range(2000):
        a = rng.uniform(-1, 1) * 10.0 ** rng.uniform(-300, 300)
        b = rng.uniform(-1, 1) * 10.0 ** rng.uniform(-300, 300)
        fa, fb = f(a), f(b)
        for got, ref in [
            (fa + fb, a + b),
            (fa - fb, a - b),
            (fa * fb, a * b),
            (fa / fb, a / b),
        ]:
            assert float(got) == ref and math.copysign(1, float(got)) == math.copysign(
                1, ref
            )


def test_special_values():
    f = core.fp23_46
    nan, inf = f.nan(), f.inf()
    assert nan.is_nan() and not (nan == nan) and nan != nan
    assert inf.is_inf() and float(inf) == math.inf
    assert float(f.inf(True)) == -math.inf
    assert (inf - inf).is_nan()
    assert (f(0.0) / f(0.0)).is_nan()
    assert float(f(1.0) / f(0.0)) == math.inf
    assert (f(1.0) * f(0.0) == f(0.0)) and not f(0.0).signbit()
    assert f(-0.0).signbit() and f(-0.0) == f(0.0)  # -0 == +0
    assert float(f()) == 0.0  # default constructs +0


def test_mixed_python_arithmetic():
    f = core.fp23_46
    x = f(2.0)
    assert float(x + 1.0) == 3.0
    assert float(2.0 * x) == 4.0
    assert x > 1.5 and x == 2.0
    assert float(abs(f(-3.0))) == 3.0
    assert bool(x) and not bool(f(0.0))


def test_repr_and_bits():
    assert core.fp8_24(1.0).bits == "0x3f800000"
    assert core.fp11_53(1.0).bits == "0x3ff0000000000000"
    assert repr(core.fp23_46(0.5)) == "fp23_46(0.5)"
    assert core.fp23_46.nbytes == 9
    assert core.fp23_46.exp_bits == 23 and core.fp23_46.precision == 46


def test_packed_array_roundtrip():
    a = np.array([1.0, -0.5, 1 / 3, 1e300, -0.0, np.inf])
    for cls, lossless in [(core.fp11_53, True), (core.fp23_46, False)]:
        packed = cls.from_float64(a)
        assert len(packed) == a.size * cls.nbytes
        back = cls.to_float64(packed)
        if lossless:
            assert np.array_equal(back, a)
        else:
            assert np.allclose(back, a, rtol=1e-13) and back[-1] == np.inf
            assert str(back[-2]) == "-0.0"  # sign of zero survives


def test_elementals_against_libm():
    """Correctly-rounded elementals are within 1 ulp of (faithful) libm."""
    f = core.fp11_53
    for x in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
        for name in ["sin", "cos", "tan", "exp", "log", "sqrt", "tanh"]:
            got = float(getattr(core, name)(f(x)))
            ref = getattr(math, name)(x)
            assert got == ref or abs(got - ref) <= abs(math.ulp(ref)), (name, x)
    assert float(core.pow(f(2.0), f(10.0))) == 1024.0
    assert float(core.atan2(f(1.0), f(1.0))) == math.atan2(1.0, 1.0)


def test_elementals_reduced_precision():
    """Elementals round to the target precision, not double's."""
    got = float(core.sin(core.fp8_8(1.0)))  # bfloat16: 8-bit precision
    assert got != math.sin(1.0)
    assert abs(got - math.sin(1.0)) <= 2.0**-8


def test_integral_rounding():
    f = core.fp23_46
    for x in [2.5, -2.5, 3.5, -0.7, 0.2, 1e30]:
        stored = float(f(x))  # x rounded to 46 bits; exact as a double
        assert float(core.trunc(f(x))) == float(math.trunc(stored))
        assert float(core.floor(f(x))) == float(math.floor(stored))
        assert float(core.ceil(f(x))) == float(math.ceil(stored))
    assert float(core.round(f(2.5))) == 3.0  # half away from zero
    assert float(core.round(f(-2.5))) == -3.0
    assert float(core.min(f(1.0), f(2.0))) == 1.0
    assert float(core.max(f(1.0), f(2.0))) == 2.0
    # dace::math semantics ((a < b) ? a : b): the second operand survives an
    # unordered compare.
    nan, one = f.nan(), f(1.0)
    assert core.min(nan, one) == one and core.min(one, nan).is_nan()
    assert core.max(nan, one) == one and core.max(one, nan).is_nan()


def test_exponent_range_beyond_double():
    """fp23_46 represents magnitudes far outside double's range via elementals."""
    f = core.fp23_46
    big = core.exp(f(1000000.0))  # e^1e6 ~ 10^434294, fits in 23 exponent bits
    assert big.is_finite() and not big.is_nan()
    assert float(big) == math.inf  # too large for a double
    assert float(core.log(big)) == 1000000.0


if __name__ == "__main__":
    test_fraction_reference_fp23_46()
    test_float32_bit_exact()
    test_float64_bit_exact()
    test_special_values()
    test_mixed_python_arithmetic()
    test_repr_and_bits()
    test_packed_array_roundtrip()
    test_elementals_against_libm()
    test_elementals_reduced_precision()
    test_integral_rounding()
    test_exponent_range_beyond_double()
    print("All fp binding tests passed.")
