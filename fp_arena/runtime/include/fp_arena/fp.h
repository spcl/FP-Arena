// Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// fp_arena::fp<Exp, Prec> -- emulated IEEE-754-style float with Exp exponent
// bits and Prec significand bits.
//
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <type_traits>

#define MPFR_USE_INTMAX_T
#include <stdint.h>

#include <mpfr.h>

namespace fp_arena {

// A value type that can be stored, copied and moved as raw bytes: trivially
// copyable, no padding, nothrow copy/destroy. Every fp<Exp, Prec> satisfies it.
template <typename T>
concept PackedValue =
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> &&
    std::has_unique_object_representations_v<T> &&
    std::is_nothrow_copy_constructible_v<T> &&
    std::is_nothrow_copy_assignable_v<T> && std::is_nothrow_destructible_v<T> &&
    std::is_trivially_destructible_v<T>;

namespace detail {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

// Number classes produced by unpack().
inline constexpr int kFinite = 0; // includes zero (mant == 0)
inline constexpr int kInf = 1;
inline constexpr int kNaN = 2;

template <typename U> inline constexpr int bit_width_any(U v) noexcept {
  if constexpr (sizeof(U) > 8) {
    const u64 hi = static_cast<u64>(v >> 64);
    return hi ? 64 + std::bit_width(hi) : std::bit_width(static_cast<u64>(v));
  } else {
    return std::bit_width(v);
  }
}

// v >> n, with shifted-out bits ORed ("jammed") into the result LSB (sticky).
template <typename U>
inline constexpr U shift_right_jam(U v, std::int64_t n) noexcept {
  if (n <= 0)
    return v;
  if (n >= static_cast<int>(sizeof(U) * 8))
    return v != 0 ? 1 : 0;
  const U lost = v & ((U(1) << n) - 1);
  return (v >> n) | (lost != 0 ? 1 : 0);
}

// Scoped widening of MPFR's exponent range to the defaults; restores the
// caller's range on exit.
struct exp_range_saver {
  mpfr_exp_t emin, emax;
  exp_range_saver() noexcept : emin(mpfr_get_emin()), emax(mpfr_get_emax()) {
    mpfr_set_emin(MPFR_EMIN_DEFAULT);
    mpfr_set_emax(MPFR_EMAX_DEFAULT);
  }
  ~exp_range_saver() {
    mpfr_set_emin(emin);
    mpfr_set_emax(emax);
  }
  exp_range_saver(const exp_range_saver &) = delete;
  exp_range_saver &operator=(const exp_range_saver &) = delete;
};

} // namespace detail

// The mpfr-backed unary elementals; drives the fp_arena free functions, the
// dace::math overloads and the nanobind bindings. X(name, mpfr_function)
#define FP_ARENA_UNARY_ELEMENTALS(X)                                           \
  X(sqrt, mpfr_sqrt)                                                           \
  X(sin, mpfr_sin)                                                             \
  X(cos, mpfr_cos)                                                             \
  X(tan, mpfr_tan)                                                             \
  X(asin, mpfr_asin)                                                           \
  X(acos, mpfr_acos)                                                           \
  X(atan, mpfr_atan)                                                           \
  X(sinh, mpfr_sinh)                                                           \
  X(cosh, mpfr_cosh)                                                           \
  X(tanh, mpfr_tanh)                                                           \
  X(exp, mpfr_exp)                                                             \
  X(exp2, mpfr_exp2)                                                           \
  X(expm1, mpfr_expm1)                                                         \
  X(log, mpfr_log)                                                             \
  X(log2, mpfr_log2)                                                           \
  X(log10, mpfr_log10)                                                         \
  X(log1p, mpfr_log1p)

template <unsigned int Exp, unsigned int Prec> class fp {
  static_assert(2 <= Exp && Exp <= 30, "exponent bits must be in [2, 30]");
  static_assert(2 <= Prec && Prec <= 64,
                "precision must be in [2, 64]; use dace::mpfr for more");

  template <unsigned int, unsigned int> friend class fp;

  using u64 = detail::u64;
  using u128 = detail::u128;

public:
  static constexpr unsigned int exponent_bits = Exp;
  static constexpr unsigned int precision = Prec;
  //: 1 sign bit + Exp exponent bits + Prec-1 explicit mantissa bits.
  static constexpr unsigned int total_bits = Exp + Prec;
  static constexpr unsigned int storage_bytes = (total_bits + 7) / 8;

private:
  static constexpr unsigned int kMantBits = Prec - 1;
  static constexpr int kBias = (1 << (Exp - 1)) - 1;
  static constexpr int kEmax = kBias; // exponent of the largest normal
  static constexpr int kEminNormal =
      1 - kBias; // exponent of the smallest normal
  static constexpr int kQmin = kEminNormal - static_cast<int>(kMantBits);
  static constexpr u64 kExpFieldMax = (u64(1) << Exp) - 1;
  static constexpr u64 kMantMask = (u64(1) << kMantBits) - 1;
  static constexpr u64 kHiddenBit = u64(1) << kMantBits;
  static constexpr unsigned int kSignPos = total_bits - 1;
  static constexpr unsigned int kSignByte = kSignPos / 8;
  static constexpr unsigned char kSignBitInByte = 1u << (kSignPos % 8);

  //: Packed value, little-endian; bits above total_bits (if any) stay zero.
  std::array<std::byte, storage_bytes> bits_;

  static_assert(std::endian::native == std::endian::little,
                "fp<Exp, Prec> assumes a little-endian host");

  // Raw storage as one or two 64-bit words; the mantissa field always lies
  // entirely in lo (kMantBits <= 63).
  u64 load_lo() const noexcept {
    static_assert(PackedValue<fp>);
    static_assert(sizeof(fp) == storage_bytes);
    u64 lo = 0;
    std::memcpy(&lo, bits_.data(), storage_bytes < 8 ? storage_bytes : 8);
    return lo;
  }

  u64 load_hi() const noexcept {
    if constexpr (storage_bytes > 8) {
      u64 hi = 0;
      std::memcpy(&hi, bits_.data() + 8, storage_bytes - 8);
      return hi;
    } else {
      return 0;
    }
  }

  static fp from_words(u64 lo, u64 hi) noexcept {
    fp r;
    std::memcpy(r.bits_.data(), &lo, storage_bytes < 8 ? storage_bytes : 8);
    if constexpr (storage_bytes > 8)
      std::memcpy(r.bits_.data() + 8, &hi, storage_bytes - 8);
    (void)hi;
    return r;
  }

  static fp make(bool sign, u64 biased_exp, u64 mant_field) noexcept {
    u64 lo = mant_field | (biased_exp << kMantBits);
    u64 hi = 0;
    if constexpr (total_bits > 64) {
      hi = (biased_exp >> (64 - kMantBits)) | (u64(sign) << (kSignPos - 64));
    } else {
      lo |= u64(sign) << kSignPos;
    }
    return from_words(lo, hi);
  }

  u64 biased_exp_field() const noexcept {
    if constexpr (total_bits > 64)
      return ((load_lo() >> kMantBits) | (load_hi() << (64 - kMantBits))) &
             kExpFieldMax;
    else
      return (load_lo() >> kMantBits) & kExpFieldMax;
  }

  struct unpacked {
    bool sign;
    int cls;        // detail::kFinite / kInf / kNaN
    std::int64_t e; // finite nonzero: value = mant * 2^(e - kMantBits)
    u64 mant;       // normalized, MSB at bit Prec-1; 0 encodes zero
  };

  unpacked unpack() const noexcept {
    const u64 lo = load_lo();
    unpacked u;
    u.e = 0;
    u.mant = 0;
    u64 biased;
    if constexpr (total_bits > 64) {
      const u64 hi = load_hi();
      u.sign = ((hi >> (kSignPos - 64)) & 1) != 0;
      biased = ((lo >> kMantBits) | (hi << (64 - kMantBits))) & kExpFieldMax;
    } else {
      u.sign = ((lo >> kSignPos) & 1) != 0;
      biased = (lo >> kMantBits) & kExpFieldMax;
    }
    const u64 mfield = lo & kMantMask;
    if (biased == kExpFieldMax) {
      u.cls = mfield == 0 ? detail::kInf : detail::kNaN;
      return u;
    }
    u.cls = detail::kFinite;
    if (biased == 0) {
      if (mfield == 0)
        return u; // +-0
      const int shift = static_cast<int>(Prec) - std::bit_width(mfield);
      u.mant = mfield << shift; // normalize the subnormal
      u.e = kEminNormal - shift;
    } else {
      u.mant = mfield | kHiddenBit;
      u.e = static_cast<std::int64_t>(biased) - kBias;
    }
    return u;
  }

  //: Working type of the arithmetic paths: u64 while every intermediate
  //: (sums with 3 guard bits, folded products/quotients, significands up to
  //: 2^Prec) fits in 64 bits; u128 for Prec 61..64.
  using rp_t = std::conditional_t<(Prec <= 60), u64, u128>;

  // Round (-1)^sign * sig * 2^q0 to nearest-even and pack, handling
  // subnormals, underflow and overflow. Contract: any information already
  // lost must be jammed into sig's low bits (bit 0 acts as sticky).
  template <typename U>
  static fp round_pack(bool sign, std::int64_t q0, U sig) noexcept {
    static_assert(std::is_same_v<U, u64> || std::is_same_v<U, u128>);
    if constexpr (Prec > 60 && sizeof(U) == 8) {
      // The rounded significand can reach 2^Prec > 2^64: widen first.
      return round_pack(sign, q0, static_cast<u128>(sig));
    } else {
      constexpr int kWidth = sizeof(U) * 8;
      if (sig == 0)
        return zero(sign);
      const int msb = detail::bit_width_any(sig) - 1;
      const std::int64_t e = q0 + msb; // exponent of the leading bit
      const std::int64_t q =
          std::max<std::int64_t>(e - kMantBits, kQmin); // result quantum
      const std::int64_t sh = q - q0; // number of sig bits below the quantum
      U m;
      if (sh <= 0) {
        m = sig << -sh; // exact
      } else if (sh > msb + 1) {
        return zero(sign); // |value| < half of the smallest subnormal
      } else {
        const U kept = sh == kWidth ? 0 : sig >> sh;
        const bool rnd = ((sig >> (sh - 1)) & 1) != 0;
        const bool stk = (sig & ((U(1) << (sh - 1)) - 1)) != 0;
        m = kept + ((rnd && (stk || (kept & 1))) ? 1 : 0);
      }
      if (m == 0)
        return zero(sign);
      std::int64_t qr = q;
      if (m == (U(1) << Prec)) { // rounding carried into an extra bit
        m >>= 1;
        qr += 1;
      }
      if (m >= (U(1) << kMantBits)) { // normal
        const std::int64_t e_res = qr + kMantBits;
        if (e_res > kEmax)
          return infinity(sign);
        return make(sign, static_cast<u64>(e_res + kBias),
                    static_cast<u64>(m) & kMantMask);
      }
      return make(sign, 0, static_cast<u64>(m)); // subnormal (qr == kQmin)
    }
  }

  static fp add_sub(const fp &l, const fp &r, bool negate_r) noexcept {
    unpacked a = l.unpack(), b = r.unpack();
    b.sign ^= negate_r;
    if (a.cls == detail::kNaN || b.cls == detail::kNaN)
      return nan();
    if (a.cls == detail::kInf || b.cls == detail::kInf) {
      if (a.cls == detail::kInf && b.cls == detail::kInf)
        return a.sign == b.sign ? infinity(a.sign) : nan();
      return a.cls == detail::kInf ? infinity(a.sign) : infinity(b.sign);
    }
    if (a.mant == 0 && b.mant == 0)
      return zero(a.sign && b.sign);
    if (a.mant == 0)
      return round_pack(b.sign, b.e - kMantBits, b.mant);
    if (b.mant == 0)
      return round_pack(a.sign, a.e - kMantBits, a.mant);
    if (b.e > a.e || (b.e == a.e && b.mant > a.mant))
      std::swap(a, b);
    // 3 guard bits; the smaller operand is shift-jammed (lost bits -> sticky).
    const rp_t sig_a = rp_t(a.mant) << 3;
    const rp_t sig_b = detail::shift_right_jam(rp_t(b.mant) << 3, a.e - b.e);
    const std::int64_t q0 = a.e - kMantBits - 3;
    if (a.sign == b.sign)
      return round_pack(a.sign, q0, static_cast<rp_t>(sig_a + sig_b));
    const rp_t diff = sig_a - sig_b;
    if (diff == 0)
      return zero(false); // exact cancellation gives +0
    return round_pack(a.sign, q0, diff);
  }

  // Convert exactly into rop (whose precision must be >= Prec).
  void to_mpfr(mpfr_ptr rop) const noexcept {
    const unpacked u = unpack();
    if (u.cls == detail::kNaN) {
      mpfr_set_nan(rop);
    } else if (u.cls == detail::kInf) {
      mpfr_set_inf(rop, u.sign ? -1 : 1);
    } else if (u.mant == 0) {
      mpfr_set_zero(rop, u.sign ? -1 : 1);
    } else {
      mpfr_set_uj_2exp(rop, u.mant, u.e - static_cast<std::int64_t>(kMantBits),
                       MPFR_RNDN);
      if (u.sign)
        mpfr_neg(rop, rop, MPFR_RNDN);
    }
  }

  // Clamp x (rounded to Prec bits, ternary value t) into this format's range
  // and extract it exactly. IEEE limits in MPFR's exponent convention:
  // emax = kEmax + 1, emin = kQmin + 1. Leaves the global range at the MPFR
  // defaults; callers restore.
  static fp from_mpfr(mpfr_ptr x, int t) noexcept {
    mpfr_set_emin(static_cast<mpfr_exp_t>(kQmin) + 1);
    mpfr_set_emax(static_cast<mpfr_exp_t>(kEmax) + 1);
    t = mpfr_check_range(x, t, MPFR_RNDN);
    mpfr_subnormalize(x, t, MPFR_RNDN);
    mpfr_set_emin(MPFR_EMIN_DEFAULT);
    mpfr_set_emax(MPFR_EMAX_DEFAULT);
    if (mpfr_nan_p(x))
      return nan();
    const bool sign = mpfr_signbit(x) != 0;
    if (mpfr_inf_p(x))
      return infinity(sign);
    if (mpfr_zero_p(x))
      return zero(sign);
    const mpfr_exp_t e = mpfr_get_exp(x); // value = m * 2^e, 0.5 <= |m| < 1
    MPFR_DECL_INIT(scaled, 64);
    mpfr_abs(scaled, x, MPFR_RNDN);
    mpfr_mul_2si(scaled, scaled, 64 - e, MPFR_RNDN); // exact, in [2^63, 2^64)
    const u64 m64 = static_cast<u64>(mpfr_get_uj(scaled, MPFR_RNDZ));
    return round_pack(sign, static_cast<std::int64_t>(e) - 64, m64);
  }

  // -1 / 0 / 1 like a three-way compare; kUnordered if either side is NaN.
  static constexpr int kUnordered = 2;
  static int compare(const fp &l, const fp &r) noexcept {
    if (l.is_nan() || r.is_nan())
      return kUnordered;
    if (l.is_zero() && r.is_zero())
      return 0;
    const bool ls = l.signbit(), rs = r.signbit();
    if (ls != rs)
      return ls ? -1 : 1;
    // Same sign: for the IEEE-style encoding the magnitude order is the
    // lexicographic (hi, lo) order of the value bits (sign stripped).
    u64 lhi = l.load_hi(), rhi = r.load_hi();
    u64 llo = l.load_lo(), rlo = r.load_lo();
    if constexpr (total_bits > 64) {
      lhi &= ~(u64(1) << (kSignPos - 64));
      rhi &= ~(u64(1) << (kSignPos - 64));
    } else {
      llo &= ~(u64(1) << kSignPos);
      rlo &= ~(u64(1) << kSignPos);
    }
    const int mag = lhi != rhi ? (lhi < rhi ? -1 : 1)
                               : (llo != rlo ? (llo < rlo ? -1 : 1) : 0);
    return ls ? -mag : mag;
  }

public:
  // Trivial (uninitialized) default construction; trivial noexcept
  // copy/destroy.
  fp() = default;
  fp(const fp &) noexcept = default;
  fp &operator=(const fp &) noexcept = default;
  ~fp() = default;

  // Conversion from any other fp format, correctly rounded.
  template <unsigned int E2, unsigned int P2>
  explicit fp(const fp<E2, P2> &o) noexcept {
    const auto u = o.unpack();
    if (u.cls == detail::kNaN)
      *this = nan();
    else if (u.cls == detail::kInf)
      *this = infinity(u.sign);
    else if (u.mant == 0)
      *this = zero(u.sign);
    else
      *this =
          round_pack(u.sign, u.e - static_cast<std::int64_t>(P2 - 1), u.mant);
  }

  fp(double v) noexcept {
    if constexpr (Exp == 11 && Prec == 53) {
      std::memcpy(bits_.data(), &v, storage_bytes);
    } else {
      fp<11, 53> d;
      std::memcpy(d.bits_.data(), &v, sizeof v);
      *this = fp(d);
    }
  }

  fp(float v) noexcept {
    if constexpr (Exp == 8 && Prec == 24) {
      std::memcpy(bits_.data(), &v, storage_bytes);
    } else {
      fp<8, 24> f;
      std::memcpy(f.bits_.data(), &v, sizeof v);
      *this = fp(f);
    }
  }

  template <std::integral I> fp(I v) noexcept {
    bool neg = false;
    u64 mag;
    if constexpr (std::is_signed_v<I>) {
      neg = v < 0;
      const u64 x = static_cast<u64>(static_cast<std::int64_t>(v));
      mag = neg ? u64(0) - x : x;
    } else {
      mag = static_cast<u64>(v);
    }
    *this = round_pack(neg, 0, mag);
  }

  // Correctly rounded conversions; float converts via fp<8,24> (single
  // rounding, no detour through double).
  explicit operator double() const noexcept {
    if constexpr (Exp == 11 && Prec == 53) {
      double d;
      std::memcpy(&d, bits_.data(), storage_bytes);
      return d;
    } else {
      const fp<11, 53> d(*this);
      double r;
      std::memcpy(&r, d.bits_.data(), sizeof r);
      return r;
    }
  }

  explicit operator float() const noexcept {
    if constexpr (Exp == 8 && Prec == 24) {
      float f;
      std::memcpy(&f, bits_.data(), storage_bytes);
      return f;
    } else {
      const fp<8, 24> f(*this);
      float r;
      std::memcpy(&r, f.bits_.data(), sizeof r);
      return r;
    }
  }

  explicit operator int() const noexcept {
    return static_cast<int>(static_cast<double>(*this));
  }
  explicit operator long() const noexcept {
    return static_cast<long>(static_cast<double>(*this));
  }
  explicit operator bool() const noexcept { return !is_zero(); }

  // Special values and classification.
  static fp zero(bool sign = false) noexcept { return make(sign, 0, 0); }
  static fp infinity(bool sign = false) noexcept {
    return make(sign, kExpFieldMax, 0);
  }
  static fp nan() noexcept {
    return make(false, kExpFieldMax, u64(1) << (kMantBits - 1)); // quiet
  }

  bool is_nan() const noexcept {
    return biased_exp_field() == kExpFieldMax && (load_lo() & kMantMask) != 0;
  }
  bool is_inf() const noexcept {
    return biased_exp_field() == kExpFieldMax && (load_lo() & kMantMask) == 0;
  }
  bool is_finite() const noexcept { return biased_exp_field() != kExpFieldMax; }
  bool is_zero() const noexcept {
    return biased_exp_field() == 0 && (load_lo() & kMantMask) == 0;
  }
  bool signbit() const noexcept {
    return (bits_[kSignByte] & std::byte{kSignBitInByte}) != std::byte{0};
  }

  // Arithmetic operators (IEEE special-value semantics, round-to-nearest-even).
  friend fp operator+(const fp &l, const fp &r) noexcept {
    return add_sub(l, r, false);
  }
  friend fp operator-(const fp &l, const fp &r) noexcept {
    return add_sub(l, r, true);
  }

  friend fp operator*(const fp &l, const fp &r) noexcept {
    const unpacked a = l.unpack(), b = r.unpack();
    const bool sign = a.sign ^ b.sign;
    if (a.cls == detail::kNaN || b.cls == detail::kNaN)
      return nan();
    if (a.cls == detail::kInf || b.cls == detail::kInf) {
      if ((a.cls == detail::kFinite && a.mant == 0) ||
          (b.cls == detail::kFinite && b.mant == 0))
        return nan(); // inf * 0
      return infinity(sign);
    }
    if (a.mant == 0 || b.mant == 0)
      return zero(sign);
    const std::int64_t q0 = (a.e - kMantBits) + (b.e - kMantBits);
    if constexpr (2 * Prec <= 64) {
      return round_pack(sign, q0, a.mant * b.mant); // exact in 64 bits
    } else if constexpr (Prec <= 60) {
      // Fold the 128-bit product into 64 bits: jam the dropped low bits into
      // the LSB, which stays strictly below the round position (>= 2 bits).
      const u128 prod = u128(a.mant) * b.mant;
      constexpr int kDrop = 2 * static_cast<int>(Prec) - 64;
      const u64 folded =
          static_cast<u64>(prod >> kDrop) |
          ((static_cast<u64>(prod) & ((u64(1) << kDrop) - 1)) != 0 ? 1 : 0);
      return round_pack(sign, q0 + kDrop, folded);
    } else {
      return round_pack(sign, q0, u128(a.mant) * b.mant);
    }
  }

  friend fp operator/(const fp &l, const fp &r) noexcept {
    const unpacked a = l.unpack(), b = r.unpack();
    const bool sign = a.sign ^ b.sign;
    if (a.cls == detail::kNaN || b.cls == detail::kNaN)
      return nan();
    if (a.cls == detail::kInf)
      return b.cls == detail::kInf ? nan() : infinity(sign);
    if (b.cls == detail::kInf)
      return zero(sign);
    if (b.mant == 0)
      return a.mant == 0 ? nan() : infinity(sign); // x / 0
    if (a.mant == 0)
      return zero(sign);
    const std::int64_t q0 = (a.e - kMantBits) - (b.e - kMantBits);
    if constexpr (Prec <= 60) {
      // Pre-normalize so the quotient has exactly 64 bits; the remainder is
      // jammed into the LSB, which stays below the round position (>= 3 bits).
      const int shift = a.mant >= b.mant ? 63 : 64;
      const u128 n = u128(a.mant) << shift;
      const u64 q = static_cast<u64>(n / b.mant);
      return round_pack(sign, q0 - shift, q | u64(n % b.mant != 0));
    } else {
      // 64 quotient bits, then two more plus a sticky bit from the remainder
      // so round_pack always sees the cut inside sig.
      const u128 n = u128(a.mant) << 64;
      const u128 q = n / b.mant;
      u128 rem = n % b.mant;
      unsigned extra = 0;
      for (int i = 0; i < 2; ++i) {
        rem <<= 1;
        extra <<= 1;
        if (rem >= b.mant) {
          rem -= b.mant;
          extra |= 1;
        }
      }
      const u128 sig = (q << 3) | (extra << 1) | (rem != 0 ? 1 : 0);
      return round_pack(sign, q0 - 64 - 3, sig);
    }
  }

  fp operator-() const noexcept {
    fp r = *this;
    r.bits_[kSignByte] ^= std::byte{kSignBitInByte};
    return r;
  }

  fp &operator+=(const fp &o) noexcept { return *this = *this + o; }
  fp &operator-=(const fp &o) noexcept { return *this = *this - o; }
  fp &operator*=(const fp &o) noexcept { return *this = *this * o; }
  fp &operator/=(const fp &o) noexcept { return *this = *this / o; }

  // Comparison operators (IEEE: NaN is unordered, -0 == +0).
  friend bool operator==(const fp &l, const fp &r) noexcept {
    return compare(l, r) == 0;
  }
  friend bool operator!=(const fp &l, const fp &r) noexcept {
    const int c = compare(l, r);
    return c == kUnordered || c != 0;
  }
  friend bool operator<(const fp &l, const fp &r) noexcept {
    return compare(l, r) == -1;
  }
  friend bool operator>(const fp &l, const fp &r) noexcept {
    return compare(l, r) == 1;
  }
  friend bool operator<=(const fp &l, const fp &r) noexcept {
    const int c = compare(l, r);
    return c == -1 || c == 0;
  }
  friend bool operator>=(const fp &l, const fp &r) noexcept {
    const int c = compare(l, r);
    return c == 1 || c == 0;
  }

  friend std::ostream &operator<<(std::ostream &os, const fp &v) {
    // Printed through double, so at most 17 significant digits survive.
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.*g",
                  std::min(17, static_cast<int>(Prec * 0.30103) + 2),
                  static_cast<double>(v));
    return os << buf;
  }

  // ---- Elemental functions (correctly rounded via libmpfr) ---------------
  //
  // Inputs move into stack-allocated mpfr_t values; the MPFR call computes at
  // Prec bits under the default exponent range and from_mpfr clamps the
  // result into this format. exp_range_saver restores the caller's range.
  //
  // The exponent range is per-thread only if MPFR was built with
  // --enable-thread-safe (mpfr_buildopt_tls_p()); otherwise concurrent
  // elemental calls race on it, like set_mpfr_exponent_bits in mpfr.h.

  using mpfr_unary_fn = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
  using mpfr_binary_fn = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr,
                                 mpfr_rnd_t);

  static fp elemental(mpfr_unary_fn f, const fp &x) noexcept {
    const detail::exp_range_saver range;
    MPFR_DECL_INIT(in, 64);
    MPFR_DECL_INIT(out, Prec);
    x.to_mpfr(in);
    const int t = f(out, in, MPFR_RNDN);
    return from_mpfr(out, t);
  }

  static fp elemental(mpfr_binary_fn f, const fp &x, const fp &y) noexcept {
    const detail::exp_range_saver range;
    MPFR_DECL_INIT(in1, 64);
    MPFR_DECL_INIT(in2, 64);
    MPFR_DECL_INIT(out, Prec);
    x.to_mpfr(in1);
    y.to_mpfr(in2);
    const int t = f(out, in1, in2, MPFR_RNDN);
    return from_mpfr(out, t);
  }

  // Exact rounding to an integral value. mode: 0 = trunc, 1 = floor,
  // 2 = ceil, 3 = round (half away from zero, as std::round).
  static fp integral_value(const fp &x, int mode) noexcept {
    const unpacked u = x.unpack();
    if (u.cls != detail::kFinite || u.mant == 0)
      return x; // nan, inf, +-0
    if (u.e >= static_cast<std::int64_t>(kMantBits))
      return x; // integral
    bool up;
    if (u.e < 0) { // |x| < 1: result is +-0 or +-1
      switch (mode) {
      case 1:
        up = u.sign;
        break;
      case 2:
        up = !u.sign;
        break;
      case 3:
        up = u.e == -1;
        break; // |x| >= 0.5
      default:
        up = false;
        break;
      }
      return up ? round_pack(u.sign, 0, u64(1)) : zero(u.sign);
    }
    const int fbits = static_cast<int>(kMantBits) - static_cast<int>(u.e);
    const u64 fmask = (u64(1) << fbits) - 1;
    const u64 frac = u.mant & fmask;
    switch (mode) {
    case 1:
      up = u.sign && frac != 0;
      break;
    case 2:
      up = !u.sign && frac != 0;
      break;
    case 3:
      up = frac >= (u64(1) << (fbits - 1));
      break;
    default:
      up = false;
      break;
    }
    const u128 m = u128(u.mant & ~fmask) + (up ? u128(fmask) + 1 : 0);
    return round_pack(u.sign, u.e - kMantBits, m);
  }

  fp abs() const noexcept {
    fp r = *this;
    r.bits_[kSignByte] &=
        std::byte{static_cast<unsigned char>(~kSignBitInByte)};
    return r;
  }
};

// Compile-time guarantees: packed (no padding), trivially copyable, exact
// size; fp<8,24> and fp<11,53> mirror the hardware float/double layouts.
static_assert(PackedValue<fp<8, 24>> && PackedValue<fp<11, 53>> &&
              PackedValue<fp<23, 46>>);
static_assert(sizeof(fp<8, 24>) == sizeof(float));
static_assert(sizeof(fp<11, 53>) == sizeof(double));
static_assert(sizeof(fp<23, 46>) == 9);

// ---- Free elemental functions (usable unqualified in tasklets via ADL) ----

#define FP_ARENA_DEFINE_UNARY(name, mpfr_fn)                                   \
  template <unsigned int E, unsigned int P>                                    \
  inline fp<E, P> name(const fp<E, P> &a) noexcept {                           \
    return fp<E, P>::elemental(mpfr_fn, a);                                    \
  }
FP_ARENA_UNARY_ELEMENTALS(FP_ARENA_DEFINE_UNARY)
#undef FP_ARENA_DEFINE_UNARY

template <unsigned int E, unsigned int P>
inline fp<E, P> pow(const fp<E, P> &a, const fp<E, P> &b) noexcept {
  return fp<E, P>::elemental(mpfr_pow, a, b);
}
template <unsigned int E, unsigned int P>
inline fp<E, P> atan2(const fp<E, P> &a, const fp<E, P> &b) noexcept {
  return fp<E, P>::elemental(mpfr_atan2, a, b);
}

template <unsigned int E, unsigned int P>
inline fp<E, P> abs(const fp<E, P> &a) noexcept {
  return a.abs();
}
template <unsigned int E, unsigned int P>
inline fp<E, P> fabs(const fp<E, P> &a) noexcept {
  return a.abs();
}
// Operand selection matches dace::math::min/max ((a < b) ? a : b), so NaN
// propagation is identical to native floats in generated code.
template <unsigned int E, unsigned int P>
inline fp<E, P> min(const fp<E, P> &a, const fp<E, P> &b) noexcept {
  return a < b ? a : b;
}
template <unsigned int E, unsigned int P>
inline fp<E, P> max(const fp<E, P> &a, const fp<E, P> &b) noexcept {
  return a > b ? a : b;
}
template <unsigned int E, unsigned int P>
inline fp<E, P> trunc(const fp<E, P> &a) noexcept {
  return fp<E, P>::integral_value(a, 0);
}
template <unsigned int E, unsigned int P>
inline fp<E, P> floor(const fp<E, P> &a) noexcept {
  return fp<E, P>::integral_value(a, 1);
}
template <unsigned int E, unsigned int P>
inline fp<E, P> ceil(const fp<E, P> &a) noexcept {
  return fp<E, P>::integral_value(a, 2);
}
template <unsigned int E, unsigned int P>
inline fp<E, P> round(const fp<E, P> &a) noexcept {
  return fp<E, P>::integral_value(a, 3);
}
template <unsigned int E, unsigned int P>
inline bool isnan(const fp<E, P> &a) noexcept {
  return a.is_nan();
}
template <unsigned int E, unsigned int P>
inline bool isinf(const fp<E, P> &a) noexcept {
  return a.is_inf();
}
template <unsigned int E, unsigned int P>
inline bool isfinite(const fp<E, P> &a) noexcept {
  return a.is_finite();
}

} // namespace fp_arena

// dace::math::<name> overloads for fp<E, P>.
namespace dace {
namespace math {

#define FP_ARENA_DACE_MATH_UNARY(name)                                         \
  template <unsigned int E, unsigned int P>                                    \
  inline ::fp_arena::fp<E, P> name(const ::fp_arena::fp<E, P> &a) {            \
    return ::fp_arena::name(a);                                                \
  }
#define FP_ARENA_DACE_MATH_UNARY_X(name, mpfr_fn) FP_ARENA_DACE_MATH_UNARY(name)

// The mpfr-backed elementals, driven by the same list as their definitions.
FP_ARENA_UNARY_ELEMENTALS(FP_ARENA_DACE_MATH_UNARY_X)
// The natively-implemented unaries.
FP_ARENA_DACE_MATH_UNARY(abs)
FP_ARENA_DACE_MATH_UNARY(fabs)
FP_ARENA_DACE_MATH_UNARY(trunc)
FP_ARENA_DACE_MATH_UNARY(round)
FP_ARENA_DACE_MATH_UNARY(floor)
#undef FP_ARENA_DACE_MATH_UNARY_X
#undef FP_ARENA_DACE_MATH_UNARY

// dace::math calls std::ceil "ceiling".
template <unsigned int E, unsigned int P>
inline ::fp_arena::fp<E, P> ceiling(const ::fp_arena::fp<E, P> &a) {
  return ::fp_arena::ceil(a);
}

template <unsigned int E, unsigned int P>
inline ::fp_arena::fp<E, P> pow(const ::fp_arena::fp<E, P> &a,
                                const ::fp_arena::fp<E, P> &b) {
  return ::fp_arena::pow(a, b);
}
template <unsigned int E, unsigned int P>
inline ::fp_arena::fp<E, P> min(const ::fp_arena::fp<E, P> &a,
                                const ::fp_arena::fp<E, P> &b) {
  return ::fp_arena::min(a, b);
}
template <unsigned int E, unsigned int P>
inline ::fp_arena::fp<E, P> max(const ::fp_arena::fp<E, P> &a,
                                const ::fp_arena::fp<E, P> &b) {
  return ::fp_arena::max(a, b);
}
template <unsigned int E, unsigned int P>
inline bool isnan(const ::fp_arena::fp<E, P> &a) {
  return a.is_nan();
}
template <unsigned int E, unsigned int P>
inline bool isinf(const ::fp_arena::fp<E, P> &a) {
  return a.is_inf();
}
template <unsigned int E, unsigned int P>
inline bool isfinite(const ::fp_arena::fp<E, P> &a) {
  return a.is_finite();
}

} // namespace math
} // namespace dace
