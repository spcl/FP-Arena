# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""DaCe GPU codegen: tasklets mixing float16 with float/double must compile.

On CUDA, dace::float16 is __half, so nvcc used to find both the built-in operator
and operator<op>(__half, __half) and reject the call as ambiguous; std::pow and abs
had no __half overload. These are the patterns behind the failed CloudSC fp16
builds. Fixed upstream in DaCe 2.0.0a10; this test guards against a regression.
"""

import shutil

import dace
import pytest
from dace.codegen.exceptions import CompilationError

f16, f32, f64 = dace.float16, dace.float32, dace.float64

CASES = [
    ("half < double literal", {"a": f16}, dace.bool_, "c = a < 1e-14"),
    ("float / half", {"a": f32, "b": f16}, f32, "c = a / b"),
    ("float == half", {"a": f32, "b": f16}, dace.bool_, "c = a == b"),
    ("pow(half, double)", {"a": f16, "b": f64}, f16, "c = a ** b"),
    ("pow(half, half)", {"a": f16, "b": f16}, f16, "c = a ** b"),
    ("abs(half)", {"a": f16}, f16, "c = abs(a)"),
]


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available")
@pytest.mark.parametrize(
    "index,name,inputs,out_dtype,code",
    [(i, *case) for i, case in enumerate(CASES)],
    ids=[case[0] for case in CASES],
)
def test_half_mixed_tasklet_compiles(index, name, inputs, out_dtype, code):
    sdfg = dace.SDFG(f"half_case_{index}")
    state = sdfg.add_state()
    for conn, dtype in {**inputs, "c": out_dtype}.items():
        sdfg.add_array(conn.upper(), [64], dtype, storage=dace.StorageType.GPU_Global)
    tasklet, _, _ = state.add_mapped_tasklet(
        "compute",
        {"i": "0:64"},
        {conn: dace.Memlet(f"{conn.upper()}[i]") for conn in inputs},
        code,
        {"c": dace.Memlet("C[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    for conn, dtype in inputs.items():
        tasklet.in_connectors[conn] = dtype
    tasklet.out_connectors["c"] = out_dtype

    try:
        sdfg.compile()
    except CompilationError as e:
        pytest.fail(f"{name}: FAILS\n{e!s}")
