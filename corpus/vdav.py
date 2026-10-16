# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
The vertical advection kernel.
Adapted from: https://github.com/spcl/npbench/tree/main
"""

import dace as dc
import numpy as np

I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ("I", "J", "K"))

dtr_stage = 3.0 / 20.0


@dc.program
def vadv(
    utens_stage: dc.float64[I, J, K],
    u_stage: dc.float64[I, J, K],
    wcon: dc.float64[I + 1, J, K],
    u_pos: dc.float64[I, J, K],
    utens: dc.float64[I, J, K],
):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * 0.5

        ccol[:, :, k] = gcv * 0.5
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (
            dtr_stage * u_pos[:, :, k]
            + utens[:, :, k]
            + utens_stage[:, :, k]
            + correction_term
        )

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * 0.5
        cs[:] = gcv * 0.5

        acol = gav * 0.5
        ccol[:, :, k] = gcv * 0.5
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (
            u_stage[:, :, k + 1] - u_stage[:, :, k]
        )
        dcol[:, :, k] = (
            dtr_stage * u_pos[:, :, k]
            + utens[:, :, k]
            + utens_stage[:, :, k]
            + correction_term
        )

        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_[:] = gav * 0.5
        acol[:] = gav * 0.5
        bcol[:] = dtr_stage - acol

        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (
            dtr_stage * u_pos[:, :, k]
            + utens[:, :, k]
            + utens_stage[:, :, k]
            + correction_term
        )

        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        # datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
