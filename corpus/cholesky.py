# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
The Cholesky decomposition kernel.
"""

import dace as dc
import numpy as np

N = dc.symbol("N", dtype=dc.int64)


@dc.program
def cholesky_kernel(A: dc.float64[N, N]):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])
