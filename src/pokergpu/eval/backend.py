from __future__ import annotations

from .cpu_stub import CpuStubLeafEvaluator
from .device import EvalDeviceConfig
from .gpu_stub import GpuStubLeafEvaluator
from .interface import LeafEvaluator


def make_leaf_evaluator(
    device_config: EvalDeviceConfig | None = None,
) -> LeafEvaluator:
    config = device_config or EvalDeviceConfig()
    if config.mode.lower() == "cpu":
        return CpuStubLeafEvaluator()
    return GpuStubLeafEvaluator(config)
