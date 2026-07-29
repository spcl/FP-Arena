"""Tests for the nanobind fp<Exp, Prec> value classes (fp_arena.native)."""

import math
import random
from fractions import Fraction

import numpy as np

import fp_arena

core = fp_arena.native


# -- an independent model of fp<Exp, Prec> -----------------------------------
#
# Values are modelled as either _NAN or a (sign, magnitude) pair, with sign in
# {-1, 1} and magnitude a non-negative Fraction or math.inf, so signed zeros
# survive. Exact rational arithmetic keeps the model independent of the C++
# implementation's guard/sticky machinery.
#
# Caution: magnitudes are exact rationals, so the value-level helpers must not
# be handed the extreme codes of a huge-exponent format -- Fraction(2)**-2**29,
# for a subnormal fp<30,4>, would need ~161 million digits and exhaust memory
# before anything failed. _pow2 refuses those outright, _classify is field-only
# and safe everywhere, and the value tests keep operands in double range.

_NAN = object()

#: Largest power-of-two exponent the model will materialize as a Fraction.
#: Comfortably covers every double-range value and every fp<E,P> with E <= 15.
_MAX_MODEL_EXP = 1 << 15


def _pow2(k: int) -> Fraction:
    """``Fraction(2)**k``, refusing exponents that would exhaust memory."""
    assert abs(k) <= _MAX_MODEL_EXP, f"model exponent {k} is out of usable range"
    return Fraction(2) ** k


def _spec(exp_bits: int, prec: int):
    """:returns: (bias, mantissa field bits, quantum exponent) of the format."""
    mant_bits = prec - 1
    bias = 2 ** (exp_bits - 1) - 1 if exp_bits else 0
    return bias, mant_bits, (1 - bias) - mant_bits


def _classify(code: int, exp_bits: int, prec: int) -> str:
    """Field-only classification of a raw code: nan / inf / zero / finite."""
    _, mant_bits, _ = _spec(exp_bits, prec)
    mfield = code & ((1 << mant_bits) - 1)
    if exp_bits == 0:
        if mfield == (1 << mant_bits) - 1:
            return "nan"
        return "inf" if mfield == (1 << mant_bits) - 2 else (
            "zero" if mfield == 0 else "finite"
        )
    efield = (code >> mant_bits) & ((1 << exp_bits) - 1)
    if efield == (1 << exp_bits) - 1:
        return "nan" if mfield else "inf"
    return "zero" if efield == 0 and mfield == 0 else "finite"


def _decode(code: int, exp_bits: int, prec: int):
    """Model of the encoding: raw code -> _NAN or (sign, magnitude)."""
    bias, mant_bits, qmin = _spec(exp_bits, prec)
    sign = -1 if (code >> (exp_bits + prec - 1)) & 1 else 1
    mfield = code & ((1 << mant_bits) - 1)
    kind = _classify(code, exp_bits, prec)
    if kind == "nan":
        return _NAN
    if kind == "inf":
        return (sign, math.inf)
    if kind == "zero":
        return (sign, Fraction(0))  # before any 2**qmin, which can be enormous
    if exp_bits == 0:
        return (sign, Fraction(mfield) * _pow2(qmin))
    efield = (code >> mant_bits) & ((1 << exp_bits) - 1)
    if efield == 0:  # zero or subnormal
        return (sign, Fraction(mfield) * _pow2(qmin))
    sig = (1 << mant_bits) | mfield
    return (sign, Fraction(sig) * _pow2(efield - bias - mant_bits))


def _encode(v, exp_bits: int, prec: int) -> int:
    """Canonical encoding of an already-representable model value."""
    bias, mant_bits, qmin = _spec(exp_bits, prec)
    if v is _NAN:
        if exp_bits == 0:
            return (1 << mant_bits) - 1
        return (((1 << exp_bits) - 1) << mant_bits) | (1 << (mant_bits - 1))
    sign, mag = v
    sbit = (1 if sign < 0 else 0) << (exp_bits + prec - 1)
    if mag == math.inf:
        if exp_bits == 0:
            return sbit | ((1 << mant_bits) - 2)
        return sbit | (((1 << exp_bits) - 1) << mant_bits)
    if mag == 0:
        return sbit
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if mag < _pow2(e):
        e -= 1
    if exp_bits and e >= 1 - bias:  # normal
        sig = mag / _pow2(e - mant_bits)
        assert sig.denominator == 1 and (1 << mant_bits) <= sig < (1 << prec)
        return sbit | ((e + bias) << mant_bits) | (int(sig) - (1 << mant_bits))
    m = mag / _pow2(qmin)  # subnormal, or the fixed grid at Exp <= 1
    assert m.denominator == 1 and m < (1 << mant_bits)
    return sbit | int(m)


def _round(sign: int, mag, exp_bits: int, prec: int):
    """Round a non-negative exact magnitude into the format, RNE."""
    bias, mant_bits, qmin = _spec(exp_bits, prec)
    if mag == math.inf:
        return (sign, math.inf)
    mag = Fraction(mag)
    if mag == 0:
        return (sign, Fraction(0))
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if mag < _pow2(e):
        e -= 1
    q = max(e - mant_bits, qmin)
    scaled = mag / _pow2(q)
    m = int(scaled)
    frac = scaled - m
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and m % 2 == 1):
        m += 1
    if m == 0:
        return (sign, Fraction(0))
    if exp_bits == 0:
        # The two largest magnitude codes are reserved, so reaching them is
        # overflow rather than a representable value.
        if m * _pow2(q) >= (2**mant_bits - 2) * _pow2(qmin):
            return (sign, math.inf)
    elif m.bit_length() - 1 + q > bias:
        return (sign, math.inf)
    return (sign, m * _pow2(q))


def _round_fp(x: Fraction, exp_bits: int, prec: int):
    """Round-to-nearest-even of an exact rational into the fp format.

    :returns: the rounded value as a Fraction, or +-inf on overflow.
    """
    sign = -1 if x < 0 else 1
    _, mag = _round(sign, abs(Fraction(x)), exp_bits, prec)
    return sign * math.inf if mag == math.inf else sign * mag


def _model_op(op: str, a, b, exp_bits: int, prec: int):
    """IEEE result of ``a op b`` on model values, rounded into the format."""
    if a is _NAN or b is _NAN:
        return _NAN
    (sa, ma), (sb, mb) = a, b
    if op == "-":
        sb, op = -sb, "+"
    inf_a, inf_b = ma == math.inf, mb == math.inf
    if op == "+":
        if inf_a and inf_b:
            return _NAN if sa != sb else (sa, math.inf)
        if inf_a or inf_b:
            return (sa, math.inf) if inf_a else (sb, math.inf)
        exact = sa * ma + sb * mb
        if exact == 0:
            # Exact cancellation gives +0; -0 only when both addends are -0.
            both_neg_zero = ma == 0 and mb == 0 and sa == sb == -1
            return (-1 if both_neg_zero else 1, Fraction(0))
        return _round(1 if exact > 0 else -1, abs(exact), exp_bits, prec)
    sign = sa * sb
    if op == "*":
        if (inf_a and mb == 0) or (inf_b and ma == 0):
            return _NAN
        if inf_a or inf_b:
            return (sign, math.inf)
        return _round(sign, ma * mb, exp_bits, prec)
    if op == "/":
        if (inf_a and inf_b) or (ma == 0 and mb == 0):
            return _NAN
        if inf_a or mb == 0:
            return (sign, math.inf)
        if inf_b or ma == 0:
            return (sign, Fraction(0))
        return _round(sign, ma / mb, exp_bits, prec)
    raise AssertionError(op)


def _assert_matches(got, expect, exp_bits: int, prec: int, ctx):
    """Assert a binding value equals a model value, comparing raw encodings."""
    if expect is _NAN:
        assert got.is_nan(), (ctx, "expected NaN")
        return
    assert not got.is_nan(), (ctx, "unexpected NaN")
    assert got.is_inf() == (expect[1] == math.inf), ctx
    assert got.is_finite() == (expect[1] != math.inf), ctx
    assert int(got.bits, 16) == _encode(expect, exp_bits, prec), (
        ctx,
        got.bits,
        hex(_encode(expect, exp_bits, prec)),
    )


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


#: The two fixed-point widths and the largest magnitude code each can hold.
#: fp<0,P> reserves its top two codes for inf/NaN, fp<1,P> reserves an
#: exponent field value instead and so keeps two more magnitudes.
_TINY = [(core.fp0_8, 0, 2**7 - 3), (core.fp1_8, 1, 2**7 - 1)]


def test_tiny_format_grid():
    """fp<0,P> and fp<1,P> are fixed-point grids of quantum 2^(2-P)."""
    for cls, exp_bits, max_m in _TINY:
        assert cls.exp_bits == exp_bits
        quantum = 2.0 ** (2 - cls.precision)
        assert float(cls(quantum)) == quantum  # smallest positive
        assert float(cls(quantum / 2)) == 0.0  # tie -> even -> 0
        assert float(cls(quantum * 1.5)) == quantum * 2  # tie -> even -> up
        assert float(cls(quantum * 2.5)) == quantum * 2  # tie -> even -> down
        max_finite = max_m * quantum
        assert float(cls(max_finite)) == max_finite
        assert cls(max_finite).is_finite()
        # Rounding past the largest finite magnitude overflows, in both
        # directions, whether it happens on construction or in an operation.
        assert (cls(max_finite) + cls(quantum)).is_inf()
        assert cls(max_finite + quantum).is_inf()
        assert (-cls(max_finite) - cls(quantum)).is_inf()
        assert (-cls(max_finite) - cls(quantum)).signbit()
        # ...but a value that still rounds back down does not.
        assert float(cls(max_finite + quantum / 4)) == max_finite


def test_tiny_format_specials():
    """The fixed-point widths keep full IEEE special-value behavior."""
    for cls, _, max_m in _TINY:
        max_finite = max_m * 2.0 ** (2 - cls.precision)
        assert cls.nan().is_nan() and cls.nan() != cls.nan()
        assert cls.inf().is_inf() and float(cls.inf()) == math.inf
        assert float(cls.inf(True)) == -math.inf
        assert (cls.inf() - cls.inf()).is_nan()
        assert (cls(0.0) / cls(0.0)).is_nan()
        assert float(cls(1.0) / cls(0.0)) == math.inf
        assert float(cls(-1.0) / cls(0.0)) == -math.inf
        assert cls(-0.0).signbit() and cls(-0.0) == cls(0.0)
        assert cls.inf() > cls(max_finite)  # inf orders above every finite
        assert cls(max_finite) > cls(0.0) > cls(-max_finite)
        assert not (cls.nan() < cls(0.0)) and not (cls.nan() >= cls(0.0))


def test_tiny_format_against_reference():
    """fp<0,P> / fp<1,P> arithmetic matches exact rational RNE on the grid."""
    rng = np.random.default_rng(11)
    for cls, exp_bits, _ in _TINY:
        prec = cls.precision
        for _ in range(400):
            # Straddles the finite range so overflow is exercised too.
            a, b = rng.uniform(-2.5, 2.5, 2)
            fa, fb = cls(float(a)), cls(float(b))
            if fa.is_inf() or fb.is_inf():
                continue
            va, vb = Fraction(float(fa)), Fraction(float(fb))
            assert (fa < fb) == (va < vb) and (fa == fb) == (va == vb)
            ops = [(fa + fb, va + vb), (fa - fb, va - vb), (fa * fb, va * vb)]
            if vb != 0:
                ops.append((fa / fb, va / vb))
            for got, exact in ops:
                expect = _round_fp(exact, exp_bits, prec)
                if expect in (math.inf, -math.inf):
                    assert got.is_inf() and got.signbit() == (expect < 0), (a, b)
                else:
                    assert Fraction(float(got)) == expect, (a, b, float(got))


def test_tiny_format_shares_grid_with_exp1():
    """fp<0,P> is exactly fp<1,P> minus its two largest magnitudes."""
    quantum = 2.0 ** (2 - core.fp0_8.precision)
    for m in range(2**7 - 3 + 1):
        assert float(core.fp0_8(m * quantum)) == float(core.fp1_8(m * quantum))
    for m in (2**7 - 2, 2**7 - 1):  # representable at Exp=1, not at Exp=0
        assert core.fp1_8(m * quantum).is_finite()
        assert core.fp0_8(m * quantum).is_inf()


# -- sweeps over the whole bound format matrix -------------------------------

#: Every (exp_bits, precision) bound in instantiations.h.
_FORMATS = [
    (0, 3), (0, 4), (0, 5), (0, 8), (0, 16), (0, 53), (0, 64),
    (1, 2), (1, 4), (1, 8), (1, 64),
    (2, 2), (2, 4), (3, 4), (8, 4), (15, 4), (30, 4),
    (5, 11), (8, 8), (8, 24), (11, 53), (8, 64), (23, 46), (30, 64),
]  # fmt: skip

#: Formats small enough to enumerate every ordered pair of encodings.
_ENUMERABLE = [f for f in _FORMATS if sum(f) <= 6]


def _cls(exp_bits: int, prec: int):
    return getattr(core, f"fp{exp_bits}_{prec}")


def _from_code(cls, code: int):
    """Build a value from a raw integer encoding, at any storage width."""
    return cls.from_bits(code.to_bytes(cls.nbytes, "little"))


def _signed(v):
    """A model value as a signed comparable (Fraction or +-inf)."""
    sign, mag = v
    return -mag if sign < 0 else mag


def test_format_matrix_is_bound():
    """Every format in the sweep list exists with the advertised geometry."""
    for exp_bits, prec in _FORMATS:
        cls = _cls(exp_bits, prec)
        assert cls.exp_bits == exp_bits and cls.precision == prec
        assert cls.nbytes == (exp_bits + prec + 7) // 8, (exp_bits, prec)


def test_encoding_classification():
    """Every code point classifies exactly as the field layout says.

    Exhaustive where the format fits in 16 bits, sampled above that. This is
    field-only, so it covers the huge-exponent formats too.
    """
    rng = random.Random(21)  # getrandbits: codes wider than 64 bits
    for exp_bits, prec in _FORMATS:
        cls = _cls(exp_bits, prec)
        total = exp_bits + prec
        codes = (
            range(1 << total)
            if total <= 16
            else [rng.getrandbits(total) for _ in range(2000)]
        )
        for code in codes:
            v = _from_code(cls, code)
            kind = _classify(code, exp_bits, prec)
            ctx = (exp_bits, prec, hex(code), kind)
            assert v.is_nan() == (kind == "nan"), ctx
            assert v.is_inf() == (kind == "inf"), ctx
            assert v.is_finite() == (kind in ("zero", "finite")), ctx
            if kind != "nan":
                assert v.signbit() == bool((code >> (total - 1)) & 1), ctx
            assert int(v.bits, 16) == code, ctx  # storage round-trips


def test_canonical_special_encodings():
    """nan()/inf() produce the encodings the model expects, at every width."""
    for exp_bits, prec in _FORMATS:
        cls = _cls(exp_bits, prec)
        assert int(cls.nan().bits, 16) == _encode(_NAN, exp_bits, prec)
        assert int(cls.inf().bits, 16) == _encode((1, math.inf), exp_bits, prec)
        assert int(cls.inf(True).bits, 16) == _encode((-1, math.inf), exp_bits, prec)
        assert int(cls().bits, 16) == _encode((1, Fraction(0)), exp_bits, prec)


def test_exhaustive_binary_ops():
    """All four ops and all six comparisons over *every ordered pair of
    encodings*, for the formats small enough to enumerate.

    This is the strongest check in the suite: it covers NaN, both infinities,
    both zeros and every subnormal, against exact rational arithmetic.
    """
    for exp_bits, prec in _ENUMERABLE:
        cls = _cls(exp_bits, prec)
        vals = [
            (code, _from_code(cls, code), _decode(code, exp_bits, prec))
            for code in range(1 << (exp_bits + prec))
        ]
        for ca, fa, ma in vals:
            for cb, fb, mb in vals:
                ctx = (exp_bits, prec, hex(ca), hex(cb))
                for op, got in (
                    ("+", fa + fb),
                    ("-", fa - fb),
                    ("*", fa * fb),
                    ("/", fa / fb),
                ):
                    expect = _model_op(op, ma, mb, exp_bits, prec)
                    _assert_matches(got, expect, exp_bits, prec, ctx + (op,))
                if ma is _NAN or mb is _NAN:  # unordered: only != is true
                    assert not (fa < fb) and not (fa > fb), ctx
                    assert not (fa <= fb) and not (fa >= fb), ctx
                    assert not (fa == fb) and (fa != fb), ctx
                else:
                    xa, xb = _signed(ma), _signed(mb)
                    assert (fa < fb) == (xa < xb), ctx
                    assert (fa > fb) == (xa > xb), ctx
                    assert (fa <= fb) == (xa <= xb), ctx
                    assert (fa >= fb) == (xa >= xb), ctx
                    assert (fa == fb) == (xa == xb), ctx
                    assert (fa != fb) == (xa != xb), ctx


def test_random_ops_across_formats():
    """Random operands through every bound format, against the model.

    Operands stay in double range so the model's exact rationals stay small
    even for the huge-exponent formats.
    """
    rng = np.random.default_rng(22)
    for exp_bits, prec in _FORMATS:
        cls = _cls(exp_bits, prec)
        for _ in range(150):
            a, b = (
                float(rng.uniform(-1, 1) * 10.0 ** rng.uniform(-12, 12))
                for _ in range(2)
            )
            fa, fb = cls(a), cls(b)
            ma = _decode(int(fa.bits, 16), exp_bits, prec)
            mb = _decode(int(fb.bits, 16), exp_bits, prec)
            for op, got in (
                ("+", fa + fb),
                ("-", fa - fb),
                ("*", fa * fb),
                ("/", fa / fb),
            ):
                expect = _model_op(op, ma, mb, exp_bits, prec)
                _assert_matches(got, expect, exp_bits, prec, (exp_bits, prec, a, b, op))


def test_format_boundaries():
    """Largest finite magnitude, first code past it, and the smallest positive
    value, checked through raw encodings so huge-exponent formats stay cheap."""
    for exp_bits, prec in _FORMATS:
        cls = _cls(exp_bits, prec)
        _, mant_bits, _ = _spec(exp_bits, prec)
        if exp_bits == 0:
            max_code, inf_code = (1 << mant_bits) - 3, (1 << mant_bits) - 2
        else:
            inf_code = ((1 << exp_bits) - 1) << mant_bits
            max_code = inf_code - 1
        ctx = (exp_bits, prec)
        biggest, beyond = _from_code(cls, max_code), _from_code(cls, inf_code)
        assert biggest.is_finite() and beyond.is_inf(), ctx
        assert _from_code(cls, inf_code + 1).is_nan(), ctx
        assert (biggest + biggest).is_inf(), ctx  # doubling always overflows
        assert (-biggest - biggest).is_inf() and (-biggest - biggest).signbit(), ctx
        # Smallest positive magnitude, and its half rounding to even -> zero.
        tiny = _from_code(cls, 1)
        assert tiny.is_finite() and not tiny.signbit() and bool(tiny), ctx
        assert not bool(tiny * cls(0.5)), ctx
        assert (-tiny).signbit() and bool(-tiny), ctx


def test_exp0_precision_sweep():
    """fp<0,P> is the grid m * 2^(2-P) with its top two codes reserved, at
    every precision from the minimum legal 3 up to the u128 path at 64."""
    precisions = [p for e, p in _FORMATS if e == 0]
    assert precisions[0] == 3 and precisions[-1] == 64
    for prec in precisions:
        cls = _cls(0, prec)
        quantum = Fraction(1, 2 ** (prec - 2))
        max_m = 2 ** (prec - 1) - 3
        ctx = ("fp<0,%d>" % prec,)
        assert cls.nbytes == (prec + 7) // 8, ctx
        # Sample codes, keeping only the ones this precision actually has
        # (fp<0,3> has just m = 0 and 1 below the reserved pair).
        for m in sorted(
            m for m in {0, 1, 2, max_m // 2, max_m - 1, max_m} if 0 <= m <= max_m
        ):
            v = _from_code(cls, m)
            assert v.is_finite(), (ctx, m)
            # The model's value for this code, re-encoded, is the code itself.
            assert _decode(m, 0, prec) == (1, m * quantum), (ctx, m)
            assert int(v.bits, 16) == _encode((1, m * quantum), 0, prec), (ctx, m)
        assert _from_code(cls, max_m + 1).is_inf(), ctx
        assert _from_code(cls, max_m + 2).is_nan(), ctx
        # The grid is closed under addition until it overflows.
        assert (_from_code(cls, max_m) + _from_code(cls, 1)).is_inf(), ctx
        if max_m >= 2:
            one = _from_code(cls, 1)
            assert int((one + one).bits, 16) == 2, ctx
        # Every fp<0,P> value is also an fp<1,P> value (same grid, wider top).
        if (1, prec) in _FORMATS:
            wide = _cls(1, prec)
            for m in (0, 1, max_m):
                assert int(_from_code(cls, m).bits, 16) == int(
                    _from_code(wide, m).bits, 16
                ), (ctx, m)


def test_exponent_sweep_dynamic_range():
    """Across the exponent sweep at fixed precision, the largest finite value
    has leading-bit exponent == bias, so the range is flat over the two
    fixed-point widths and then grows with the bias."""
    tops = {}
    for exp_bits, prec in _FORMATS:
        if prec != 4:
            continue
        cls = _cls(exp_bits, prec)
        bias, mant_bits, _ = _spec(exp_bits, prec)
        max_code = (
            (1 << mant_bits) - 3
            if exp_bits == 0
            else (((1 << exp_bits) - 1) << mant_bits) - 1
        )
        assert _from_code(cls, max_code).is_finite(), exp_bits
        assert _from_code(cls, max_code + 1).is_inf(), exp_bits
        if exp_bits > 15:
            continue  # the exact rational here would have millions of digits
        mag = _signed(_decode(max_code, exp_bits, prec))
        e = mag.numerator.bit_length() - mag.denominator.bit_length()
        if mag < _pow2(e):
            e -= 1
        tops[exp_bits] = e
        assert e == bias, (exp_bits, e, bias)
    # Exp 0 and 1 share one range; from Exp 2 the bias, and so the range, grows.
    assert tops[0] == tops[1] == 0, tops
    assert tops[2] == 1 and tops[3] == 3, tops
    assert tops[8] == 127 and tops[15] == 16383, tops


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
    test_tiny_format_grid()
    test_tiny_format_specials()
    test_tiny_format_against_reference()
    test_tiny_format_shares_grid_with_exp1()
    test_format_matrix_is_bound()
    test_encoding_classification()
    test_canonical_special_encodings()
    test_exhaustive_binary_ops()
    test_random_ops_across_formats()
    test_format_boundaries()
    test_exp0_precision_sweep()
    test_exponent_sweep_dynamic_range()
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
