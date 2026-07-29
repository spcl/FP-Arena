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
  X(23, 46)                                                                    \
  /* Exp = 0: fixed point, no exponent field, top two codes are inf/NaN.    */ \
  X(0, 3)  /* smallest legal Exp = 0 format                             */     \
  X(0, 4)                                                                      \
  X(0, 5)                                                                      \
  X(0, 8)                                                                      \
  X(0, 16)                                                                     \
  X(0, 53)                                                                     \
  X(0, 64) /* Exp = 0 on the u128 arithmetic path                       */     \
  /* Exp = 1: fixed point, one exponent field value spent on inf/NaN.      */ \
  X(1, 2)  /* smallest legal format overall                             */     \
  X(1, 4)                                                                      \
  X(1, 8)                                                                      \
  X(1, 64)                                                                     \
  /* Exp >= 2: real binades, plus an exponent sweep at fixed precision.    */ \
  X(2, 2)                                                                      \
  X(2, 4)                                                                      \
  X(3, 4)                                                                      \
  X(8, 4)                                                                      \
  X(15, 4)                                                                     \
  X(30, 4)                                                                     \
  X(30, 64) /* widest exponent and precision together                   */
