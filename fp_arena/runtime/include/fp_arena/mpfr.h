#pragma once

#include <mpfr.h>

#include <iostream>

namespace dace {

// Process-global exponent width (0 = unlimited). Set once before any
// computation. Not thread-safe; the per-operation clamp also mutates MPFR's
// exponent range, which is per-thread only on --enable-thread-safe builds
// (mpfr_buildopt_tls_p()).
inline unsigned int mpfr_exponent_bits = 0;

// Configure the exponent width for all dace::mpfr values. Each operation is
// clamped to the IEEE-754 range of a format with that many exponent bits and
// the value's own precision. IEEE limits in MPFR's exponent convention
// (x = m*2^e, 0.5 <= |m| < 1): emax = bias+1 = 2^(bits-1),
// emin = 4 - emax - precision (the MPFR manual's IEEE-double values
// 1024 / -1073, generalized).
inline void set_mpfr_exponent_bits(unsigned int bits) {
  mpfr_exponent_bits = bits;
}

template <unsigned int Precision> class mpfr {
private:
  mpfr_t val;

  void clamp_exp(int t) {
    if (mpfr_exponent_bits > 0) {
      const mpfr_exp_t semin = mpfr_get_emin(), semax = mpfr_get_emax();
      const mpfr_exp_t emax = mpfr_exp_t(1) << (mpfr_exponent_bits - 1);
      mpfr_set_emax(emax);
      mpfr_set_emin(4 - emax - static_cast<mpfr_exp_t>(Precision));
      t = mpfr_check_range(val, t, MPFR_RNDN);
      mpfr_subnormalize(val, t, MPFR_RNDN);
      mpfr_set_emin(semin);
      mpfr_set_emax(semax);
    }
  }

public:
  // Constructors
  mpfr() { mpfr_init2(val, Precision); }

  mpfr(double d) {
    mpfr_init2(val, Precision);
    clamp_exp(mpfr_set_d(val, d, MPFR_RNDN));
  }

  mpfr(float f) {
    mpfr_init2(val, Precision);
    clamp_exp(mpfr_set_flt(val, f, MPFR_RNDN));
  }

  mpfr(int i) {
    mpfr_init2(val, Precision);
    clamp_exp(mpfr_set_si(val, i, MPFR_RNDN));
  }

  mpfr(unsigned int i) {
    mpfr_init2(val, Precision);
    clamp_exp(mpfr_set_ui(val, i, MPFR_RNDN));
  }

  mpfr(long int i) {
    mpfr_init2(val, Precision);
    clamp_exp(mpfr_set_si(val, i, MPFR_RNDN));
  }

  mpfr(unsigned long int i) {
    mpfr_init2(val, Precision);
    clamp_exp(mpfr_set_ui(val, i, MPFR_RNDN));
  }

  // Copy Constructor
  mpfr(const mpfr &other) {
    mpfr_init2(val, Precision);
    mpfr_set(val, other.val, MPFR_RNDN);
  }

  // Move Constructor
  mpfr(mpfr &&other) noexcept {
    mpfr_init2(val, Precision);
    mpfr_swap(val, other.val);
  }

  // Destructor
  ~mpfr() { mpfr_clear(val); }

  // Copy Assignment
  mpfr &operator=(const mpfr &other) {
    if (this != &other) {
      mpfr_set(val, other.val, MPFR_RNDN);
    }
    return *this;
  }

  // Move Assignment
  mpfr &operator=(mpfr &&other) noexcept {
    if (this != &other) {
      mpfr_swap(val, other.val);
    }
    return *this;
  }

  // Assign from basic types
  mpfr &operator=(double d) {
    clamp_exp(mpfr_set_d(val, d, MPFR_RNDN));
    return *this;
  }

  mpfr &operator=(float f) {
    clamp_exp(mpfr_set_flt(val, f, MPFR_RNDN));
    return *this;
  }

  mpfr &operator=(int i) {
    clamp_exp(mpfr_set_si(val, i, MPFR_RNDN));
    return *this;
  }

  // Conversions
  explicit operator double() const { return mpfr_get_d(val, MPFR_RNDN); }

  explicit operator float() const { return mpfr_get_flt(val, MPFR_RNDN); }

  explicit operator int() const { return (int)mpfr_get_si(val, MPFR_RNDN); }

  explicit operator long() const { return mpfr_get_si(val, MPFR_RNDN); }

  explicit operator bool() const { return mpfr_cmp_d(val, 0.0) != 0; }

  // Arithmetic Operators
  friend mpfr operator+(const mpfr &lhs, const mpfr &rhs) {
    mpfr result;
    result.clamp_exp(mpfr_add(result.val, lhs.val, rhs.val, MPFR_RNDN));
    return result;
  }

  friend mpfr operator-(const mpfr &lhs, const mpfr &rhs) {
    mpfr result;
    result.clamp_exp(mpfr_sub(result.val, lhs.val, rhs.val, MPFR_RNDN));
    return result;
  }

  friend mpfr operator*(const mpfr &lhs, const mpfr &rhs) {
    mpfr result;
    result.clamp_exp(mpfr_mul(result.val, lhs.val, rhs.val, MPFR_RNDN));
    return result;
  }

  friend mpfr operator/(const mpfr &lhs, const mpfr &rhs) {
    mpfr result;
    result.clamp_exp(mpfr_div(result.val, lhs.val, rhs.val, MPFR_RNDN));
    return result;
  }

  // Unary minus
  mpfr operator-() const {
    mpfr result;
    result.clamp_exp(mpfr_neg(result.val, this->val, MPFR_RNDN));
    return result;
  }

  // Compound assignments
  mpfr &operator+=(const mpfr &other) {
    clamp_exp(mpfr_add(val, val, other.val, MPFR_RNDN));
    return *this;
  }

  mpfr &operator-=(const mpfr &other) {
    clamp_exp(mpfr_sub(val, val, other.val, MPFR_RNDN));
    return *this;
  }

  mpfr &operator*=(const mpfr &other) {
    clamp_exp(mpfr_mul(val, val, other.val, MPFR_RNDN));
    return *this;
  }

  mpfr &operator/=(const mpfr &other) {
    clamp_exp(mpfr_div(val, val, other.val, MPFR_RNDN));
    return *this;
  }

  // Comparison Operators
  friend bool operator==(const mpfr &lhs, const mpfr &rhs) {
    return mpfr_cmp(lhs.val, rhs.val) == 0;
  }

  friend bool operator!=(const mpfr &lhs, const mpfr &rhs) {
    return mpfr_cmp(lhs.val, rhs.val) != 0;
  }

  friend bool operator<(const mpfr &lhs, const mpfr &rhs) {
    return mpfr_cmp(lhs.val, rhs.val) < 0;
  }

  friend bool operator>(const mpfr &lhs, const mpfr &rhs) {
    return mpfr_cmp(lhs.val, rhs.val) > 0;
  }

  friend bool operator<=(const mpfr &lhs, const mpfr &rhs) {
    return mpfr_cmp(lhs.val, rhs.val) <= 0;
  }

  friend bool operator>=(const mpfr &lhs, const mpfr &rhs) {
    return mpfr_cmp(lhs.val, rhs.val) >= 0;
  }

  // IO Stream
  friend std::ostream &operator<<(std::ostream &os, const mpfr &m) {
    char *str = nullptr;
    mpfr_asprintf(&str, "%.*RNg", (int)(Precision * 0.30103) + 2, m.val);
    if (str) {
      os << str;
      mpfr_free_str(str);
    }
    return os;
  }
};

} // namespace dace
