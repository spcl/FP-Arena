// Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
//
// fp<Exp, Prec> instantiations exposed to Python, each as a class named
// fp<Exp>_<Prec>. Append a line to add one.
#pragma once

#define FP_ARENA_FP_INSTANTIATIONS(X)                                          \
  X(5, 11)  /* IEEE half        */                                             \
  X(8, 8)   /* bfloat16         */                                             \
  X(8, 24)  /* IEEE single      */                                             \
  X(11, 53) /* IEEE double      */                                             \
  X(8, 64)                                                                     \
  X(23, 46)
