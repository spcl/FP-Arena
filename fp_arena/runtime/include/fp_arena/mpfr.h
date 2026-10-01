#pragma once

#include <mpfr.h>

#include <iostream>

namespace dace {

// Process-global exponent width (0 = unlimited).
// Set once before any computation via set_mpfr_exponent_bits(). Not
// thread-safe.
inline unsigned int mpfr_exponent_bits = 0;

// Configure the exponent range for all dace::mpfr values.
// Uses IEEE-style bias: emax = 2^(bits-1)-1, emin = 1-emax.

inline void set_mpfr_exponent_bits(unsigned int bits) {
  mpfr_exponent_bits = bits;
  if (bits > 0) {
    mpfr_exp_t emax = (mpfr_exp_t(1) << (bits - 1)) - 1;
    mpfr_set_emin(1 - emax);
    mpfr_set_emax(emax);
  } else {
    mpfr_set_emin(MPFR_EMIN_DEFAULT);
    mpfr_set_emax(MPFR_EMAX_DEFAULT);
  }
}

template <unsigned int Precision> class mpfr {
private:
  mpfr_t val;

  void clamp_exp(int t) {
    if (mpfr_exponent_bits > 0) {
      int temp = mpfr_check_range(val, t, MPFR_RNDN);
      mpfr_subnormalize(val, temp, MPFR_RNDN);
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

  // Raw MPFR handle, for the math functions below.
  mpfr_ptr raw() { return val; }
  mpfr_srcptr raw() const { return val; }

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

// Math functions for dace::mpfr.
namespace math {

#define FP_ARENA_MPFR_UNARY(NAME, MPFR_FN)                                     \
  template <unsigned int P> inline mpfr<P> NAME(const mpfr<P> &a) {            \
    mpfr<P> r;                                                                 \
    MPFR_FN(r.raw(), a.raw(), MPFR_RNDN);                                      \
    return r;                                                                  \
  }

FP_ARENA_MPFR_UNARY(exp, mpfr_exp)
FP_ARENA_MPFR_UNARY(log, mpfr_log)
FP_ARENA_MPFR_UNARY(log10, mpfr_log10)
FP_ARENA_MPFR_UNARY(sqrt, mpfr_sqrt)
FP_ARENA_MPFR_UNARY(tanh, mpfr_tanh)
FP_ARENA_MPFR_UNARY(abs, mpfr_abs)
FP_ARENA_MPFR_UNARY(fabs, mpfr_abs)

#undef FP_ARENA_MPFR_UNARY

template <unsigned int P>
inline mpfr<P> pow(const mpfr<P> &a, const mpfr<P> &b) {
  mpfr<P> r;
  mpfr_pow(r.raw(), a.raw(), b.raw(), MPFR_RNDN);
  return r;
}

template <unsigned int P> inline mpfr<P> pow(const mpfr<P> &a, const double &b) {
  return pow(a, mpfr<P>(b));
}

template <unsigned int P> inline mpfr<P> pow(const double &a, const mpfr<P> &b) {
  return pow(mpfr<P>(a), b);
}

template <unsigned int P> inline mpfr<P> pow(const mpfr<P> &a, const int &b) {
  mpfr<P> r;
  mpfr_pow_si(r.raw(), a.raw(), b, MPFR_RNDN);
  return r;
}

template <unsigned int P>
inline mpfr<P> min(const mpfr<P> &a, const mpfr<P> &b) {
  return b < a ? b : a;
}
template <unsigned int P> inline mpfr<P> min(const mpfr<P> &a, const double &b) {
  return min(a, mpfr<P>(b));
}
template <unsigned int P> inline mpfr<P> min(const double &a, const mpfr<P> &b) {
  return min(mpfr<P>(a), b);
}
template <unsigned int P>
inline mpfr<P> max(const mpfr<P> &a, const mpfr<P> &b) {
  return a < b ? b : a;
}
template <unsigned int P> inline mpfr<P> max(const mpfr<P> &a, const double &b) {
  return max(a, mpfr<P>(b));
}
template <unsigned int P> inline mpfr<P> max(const double &a, const mpfr<P> &b) {
  return max(mpfr<P>(a), b);
}

} // namespace math

template <unsigned int P> inline mpfr<P> abs(const mpfr<P> &a) {
  return math::abs(a);
}

} // namespace dace

template <unsigned int P>
inline dace::mpfr<P> Min(const dace::mpfr<P> &a, const dace::mpfr<P> &b) {
  return dace::math::min(a, b);
}
template <unsigned int P>
inline dace::mpfr<P> Min(const dace::mpfr<P> &a, const double &b) {
  return dace::math::min(a, b);
}
template <unsigned int P>
inline dace::mpfr<P> Min(const double &a, const dace::mpfr<P> &b) {
  return dace::math::min(a, b);
}
template <unsigned int P>
inline dace::mpfr<P> Max(const dace::mpfr<P> &a, const dace::mpfr<P> &b) {
  return dace::math::max(a, b);
}
template <unsigned int P>
inline dace::mpfr<P> Max(const dace::mpfr<P> &a, const double &b) {
  return dace::math::max(a, b);
}
template <unsigned int P>
inline dace::mpfr<P> Max(const double &a, const dace::mpfr<P> &b) {
  return dace::math::max(a, b);
}
