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
            board_size = tensors["board_size"].to(dtype=torch.float32)
            stack_gap = tensors["stack_p0"] - tensors["stack_p1"]
            street = tensors["street"].to(dtype=torch.float32)
            reach_delta = tensors["reach_p0"] - tensors["reach_p1"]
            terminal_bias = (~tensors["is_terminal"]).to(dtype=torch.float32) * 0.1
            ev_p0 = (
                0.0001 * pot
                + 0.01 * stack_gap
                + 0.05 * board_size
                + 0.02 * street
                + 0.5 * reach_delta
                + terminal_bias
            )
            ev_p1 = -ev_p0
            return LeafValueBatch(
                ev_player0=ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
                ev_player1=ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
            )
        except Exception:
            return self._cpu_fallback.evaluate(batch)
