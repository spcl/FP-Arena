#pragma once

#include <mpfr.h>

#include <iostream>

namespace dace {

template <unsigned int Precision>
class mpfr {
 private:
  mpfr_t val;

 public:
  // Constructors
  mpfr() { mpfr_init2(val, Precision); }

  mpfr(double d) {
    mpfr_init2(val, Precision);
    mpfr_set_d(val, d, MPFR_RNDN);
  }

  mpfr(float f) {
    mpfr_init2(val, Precision);
    mpfr_set_flt(val, f, MPFR_RNDN);
  }

  mpfr(int i) {
    mpfr_init2(val, Precision);
    mpfr_set_si(val, i, MPFR_RNDN);
  }

  mpfr(unsigned int i) {
    mpfr_init2(val, Precision);
    mpfr_set_ui(val, i, MPFR_RNDN);
  }

  mpfr(long int i) {
    mpfr_init2(val, Precision);
    mpfr_set_si(val, i, MPFR_RNDN);
  }

  mpfr(unsigned long int i) {
    mpfr_init2(val, Precision);
    mpfr_set_ui(val, i, MPFR_RNDN);
  }

  // Copy Constructor
  mpfr(const mpfr& other) {
    mpfr_init2(val, Precision);
    mpfr_set(val, other.val, MPFR_RNDN);
  }

  // Move Constructor
  mpfr(mpfr&& other) noexcept {
    mpfr_init2(val, Precision);
    mpfr_swap(val, other.val);
  }

  // Destructor
  ~mpfr() { mpfr_clear(val); }

  // Copy Assignment
  mpfr& operator=(const mpfr& other) {
    if (this != &other) {
      mpfr_set(val, other.val, MPFR_RNDN);
    }
    return *this;
  }

  // Move Assignment
  mpfr& operator=(mpfr&& other) noexcept {
    if (this != &other) {
      mpfr_swap(val, other.val);
    }
    return *this;
  }

  // Assign from basic types
  mpfr& operator=(double d) {
    mpfr_set_d(val, d, MPFR_RNDN);
    return *this;
  }

  mpfr& operator=(float f) {
    mpfr_set_flt(val, f, MPFR_RNDN);
    return *this;
  }

  mpfr& operator=(int i) {
    mpfr_set_si(val, i, MPFR_RNDN);
    return *this;
  }

  // Conversions
  explicit operator double() const { return mpfr_get_d(val, MPFR_RNDN); }

  explicit operator float() const { return mpfr_get_flt(val, MPFR_RNDN); }

  explicit operator int() const { return (int)mpfr_get_si(val, MPFR_RNDN); }

  explicit operator long() const { return mpfr_get_si(val, MPFR_RNDN); }

  explicit operator bool() const { return mpfr_cmp_d(val, 0.0) != 0; }

  // Arithmetic Operators
  friend mpfr operator+(const mpfr& lhs, const mpfr& rhs) {
    mpfr result;
    mpfr_add(result.val, lhs.val, rhs.val, MPFR_RNDN);
    return result;
  }

  friend mpfr operator-(const mpfr& lhs, const mpfr& rhs) {
    mpfr result;
    mpfr_sub(result.val, lhs.val, rhs.val, MPFR_RNDN);
    return result;
  }

  friend mpfr operator*(const mpfr& lhs, const mpfr& rhs) {
    mpfr result;
    mpfr_mul(result.val, lhs.val, rhs.val, MPFR_RNDN);
    return result;
  }

  friend mpfr operator/(const mpfr& lhs, const mpfr& rhs) {
    mpfr result;
    mpfr_div(result.val, lhs.val, rhs.val, MPFR_RNDN);
    return result;
  }

  // Unary minus
  mpfr operator-() const {
    mpfr result;
    mpfr_neg(result.val, this->val, MPFR_RNDN);
    return result;
  }

  // Compound assignments
  mpfr& operator+=(const mpfr& other) {
    mpfr_add(val, val, other.val, MPFR_RNDN);
    return *this;
  }

  mpfr& operator-=(const mpfr& other) {
    mpfr_sub(val, val, other.val, MPFR_RNDN);
    return *this;
  }

  mpfr& operator*=(const mpfr& other) {
    mpfr_mul(val, val, other.val, MPFR_RNDN);
    return *this;
  }

  mpfr& operator/=(const mpfr& other) {
    mpfr_div(val, val, other.val, MPFR_RNDN);
    return *this;
  }

  // Comparison Operators
  friend bool operator==(const mpfr& lhs, const mpfr& rhs) {
    return mpfr_cmp(lhs.val, rhs.val) == 0;
  }

  friend bool operator!=(const mpfr& lhs, const mpfr& rhs) {
    return mpfr_cmp(lhs.val, rhs.val) != 0;
  }

  friend bool operator<(const mpfr& lhs, const mpfr& rhs) {
    return mpfr_cmp(lhs.val, rhs.val) < 0;
  }

  friend bool operator>(const mpfr& lhs, const mpfr& rhs) {
    return mpfr_cmp(lhs.val, rhs.val) > 0;
  }

  friend bool operator<=(const mpfr& lhs, const mpfr& rhs) {
    return mpfr_cmp(lhs.val, rhs.val) <= 0;
  }

  friend bool operator>=(const mpfr& lhs, const mpfr& rhs) {
    return mpfr_cmp(lhs.val, rhs.val) >= 0;
  }

  // IO Stream
  friend std::ostream& operator<<(std::ostream& os, const mpfr& m) {
    char* str = nullptr;
    mpfr_asprintf(&str, "%.*RNg", (int)(Precision * 0.30103) + 2, m.val);
    if (str) {
      os << str;
      mpfr_free_str(str);
    }
    return os;
  }
};

}  // namespace dace
