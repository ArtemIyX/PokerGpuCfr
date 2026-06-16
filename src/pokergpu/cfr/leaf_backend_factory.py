from __future__ import annotations

from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.gpu_leaf_backend import TorchLeafKernel
from pokergpu.cfr.leaf_model_spec import GpuLeafModelSpec
from pokergpu.cfr.triton_leaf_backend import TritonLeafKernel


def create_leaf_backend(
    *,
    spec: GpuLeafModelSpec = GpuLeafModelSpec(),
    prefer_triton: bool = False,
) -> GpuLeafBackend:
    if prefer_triton:
        return GpuLeafBackend(kernel=TritonLeafKernel(spec=spec), spec=spec)
    return GpuLeafBackend(kernel=TorchLeafKernel(spec=spec), spec=spec)
