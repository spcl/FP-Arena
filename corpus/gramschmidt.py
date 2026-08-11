# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
The Gram-Schmidt orthogonalization kernel.
"""

import dace as dc
import numpy as np

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


@dc.program
def gramschmidt_kernel(A: dc.float64[M, N], Q: dc.float64[M, N], R: dc.float64[N, N]):

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]
