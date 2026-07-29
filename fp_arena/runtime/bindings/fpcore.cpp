// Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// nanobind bindings for fp_arena::fp<Exp, Prec>: one Python class per
// instantiation in instantiations.h, named like the DaCe dtype strings
// (fp23_46).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <sstream>
#include <string>

#include <fp_arena/fp.h>

#include "instantiations.h"

namespace nb = nanobind;

namespace {

template <unsigned int E, unsigned int P> void bind_fp(nb::module_ &m) {
  using T = fp_arena::fp<E, P>;
  static_assert(fp_arena::PackedValue<T>);
  const std::string name = "fp" + std::to_string(E) + "_" + std::to_string(P);

  auto cls =
      nb::class_<T>(m, name.c_str())
          .def("__init__", [](T *t) { new (t) T(T::zero()); })
          .def(nb::init<double>())
          .def(nb::self + nb::self)
          .def(nb::self - nb::self)
          .def(nb::self * nb::self)
          .def(nb::self / nb::self)
          // reflected forms so `2.0 * x` works (Python floats on the left)
          .def(double() + nb::self)
          .def(double() - nb::self)
          .def(double() * nb::self)
          .def(double() / nb::self)
          .def(nb::self += nb::self)
          .def(nb::self -= nb::self)
          .def(nb::self *= nb::self)
          .def(nb::self /= nb::self)
          .def(-nb::self)
          .def(nb::self == nb::self)
          .def(nb::self != nb::self)
          .def(nb::self < nb::self)
          .def(nb::self <= nb::self)
          .def(nb::self > nb::self)
          .def(nb::self >= nb::self)
          .def("__float__", [](const T &v) { return double(v); })
          .def("__bool__", [](const T &v) { return bool(v); })
          .def("__abs__", [](const T &v) { return v.abs(); })
          .def("__str__",
               [](const T &v) {
                 std::ostringstream os;
                 os << v;
                 return os.str();
               })
          .def("__repr__",
               [name](const T &v) {
                 std::ostringstream os;
                 os << name << "(" << v << ")";
                 return os.str();
               })
          .def_prop_ro("bits",
                       [](const T &v) {
                         // big-endian hex string, e.g. "0x3f800000"
                         unsigned char raw[sizeof(T)];
                         std::memcpy(raw, &v, sizeof(T));
                         std::string s = "0x";
                         for (int i = sizeof(T) - 1; i >= 0; --i) {
                           char b[3];
                           std::snprintf(b, sizeof b, "%02x", raw[i]);
                           s += b;
                         }
                         return s;
                       })
          .def_static(
              "from_bits",
              [](nb::bytes raw) {
                if (raw.size() != sizeof(T))
                  throw nb::value_error("raw storage must be exactly nbytes");
                // The type assumes any storage above total_bits stays zero.
                constexpr unsigned kStray = sizeof(T) * 8 - T::total_bits;
                if constexpr (kStray > 0) {
                  const auto top =
                      static_cast<unsigned char>(raw.c_str()[sizeof(T) - 1]);
                  if (top >> (8 - kStray))
                    throw nb::value_error("bits above total_bits must be zero");
                }
                T v;
                std::memcpy(&v, raw.c_str(), sizeof(T));
                return v;
              },
              "Reinterpret raw little-endian storage bytes as a value.")
          .def("is_nan", &T::is_nan)
          .def("is_inf", &T::is_inf)
          .def("is_finite", &T::is_finite)
          .def("signbit", &T::signbit)
          .def_static("nan", &T::nan)
          .def_static("inf", &T::infinity, nb::arg("sign") = false)
          .def_static(
              "from_float64",
              [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> a) {
                std::string buf(a.shape(0) * sizeof(T), '\0');
                T *out = reinterpret_cast<T *>(buf.data());
                for (size_t i = 0; i < a.shape(0); ++i)
                  out[i] = T(a(i));
                return nb::bytes(buf.data(), buf.size());
              },
              "Convert a float64 array to packed bytes of this type.")
          .def_static(
              "to_float64",
              [](nb::bytes b) {
                const size_t n = b.size() / sizeof(T);
                if (n * sizeof(T) != b.size())
                  throw nb::value_error("buffer size is not a multiple of "
                                        "the packed element size");
                const T *in = reinterpret_cast<const T *>(b.c_str());
                double *out = new double[n];
                for (size_t i = 0; i < n; ++i)
                  out[i] = double(in[i]);
                nb::capsule owner(out, [](void *p) noexcept {
                  delete[] static_cast<double *>(p);
                });
                return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, {n},
                                                                   owner);
              },
              "Convert packed bytes of this type to a float64 array.");
  cls.attr("exp_bits") = E;
  cls.attr("precision") = P;
  cls.attr("nbytes") = sizeof(T);

  // Implicit double -> fp conversion for mixed expressions (x + 1.0).
  nb::implicitly_convertible<double, T>();

#define FP_ARENA_BIND_UNARY(fn)                                                \
  m.def(#fn, [](const T &v) { return fp_arena::fn(v); });
#define FP_ARENA_BIND_UNARY_X(fn, mpfr_fn) FP_ARENA_BIND_UNARY(fn)
  // The mpfr-backed elementals, driven by the same list as their definitions.
  FP_ARENA_UNARY_ELEMENTALS(FP_ARENA_BIND_UNARY_X)
  // The natively-implemented unaries.
  FP_ARENA_BIND_UNARY(trunc)
  FP_ARENA_BIND_UNARY(floor)
  FP_ARENA_BIND_UNARY(ceil)
  FP_ARENA_BIND_UNARY(round)
#undef FP_ARENA_BIND_UNARY_X
#undef FP_ARENA_BIND_UNARY
  m.def("pow", [](const T &a, const T &b) { return fp_arena::pow(a, b); });
  m.def("atan2", [](const T &a, const T &b) { return fp_arena::atan2(a, b); });
  m.def("min", [](const T &a, const T &b) { return fp_arena::min(a, b); });
  m.def("max", [](const T &a, const T &b) { return fp_arena::max(a, b); });
}

} // namespace

NB_MODULE(_fpcore, m) {
  m.doc() = "FP-Arena emulated floating-point value types (fp<Exp, Prec>)";
#define FP_ARENA_BIND(E, P) bind_fp<E, P>(m);
  FP_ARENA_FP_INSTANTIATIONS(FP_ARENA_BIND)
#undef FP_ARENA_BIND
}
