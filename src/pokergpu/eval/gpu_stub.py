from __future__ import annotations

import torch

from .cpu_stub import CpuStubLeafEvaluator
from .device import EvalDeviceConfig, resolve_eval_device
from .interface import LeafEvaluator
from .tensor_builder import build_gpu_leaf_tensors
from .types import LeafFeatureBatch, LeafValueBatch


class GpuStubLeafEvaluator(LeafEvaluator):
    def __init__(self, device_config: EvalDeviceConfig | None = None) -> None:
        self._device_config = device_config or EvalDeviceConfig()
        self._device = resolve_eval_device(self._device_config)
        self._cpu_fallback = CpuStubLeafEvaluator()

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        if self._device.type != "cuda":
            return self._cpu_fallback.evaluate(batch)

        try:
            tensors = build_gpu_leaf_tensors(batch, self._device)
            pot = tensors["pot"]
            reach_delta = tensors["reach_p0"] - tensors["reach_p1"]
            ev_p0 = 0.5 * pot + reach_delta
            ev_p1 = -0.5 * pot - reach_delta
            return LeafValueBatch(
                ev_player0=ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
                ev_player1=ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
            )
        except Exception:
            return self._cpu_fallback.evaluate(batch)
