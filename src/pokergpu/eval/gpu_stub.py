from __future__ import annotations

import numpy as np
import torch

from pokergpu.core.state import HandPhase

from .cpu_stub import CpuStubLeafEvaluator
from .device import EvalDeviceConfig, resolve_eval_device
from .interface import LeafEvaluator
from .tensor_builder import build_gpu_leaf_tensors
from .types import LeafFeatureBatch, LeafValueBatch


class GpuBatchLeafEvaluator(LeafEvaluator):
    def __init__(self, device_config: EvalDeviceConfig | None = None) -> None:
        self._device_config = device_config or EvalDeviceConfig()
        self._device = resolve_eval_device(self._device_config)
        self._cpu_fallback = CpuStubLeafEvaluator()

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        from pokergpu.core.payouts import compute_payouts

        if self._device.type != "cuda":
            return self._cpu_fallback.evaluate(batch)

        try:
            build_gpu_leaf_tensors(batch, self._device)
            ev_p0 = torch.zeros(batch.size, dtype=torch.float32, device=self._device)
            ev_p1 = torch.zeros(batch.size, dtype=torch.float32, device=self._device)
            terminal_mask = ~torch.isnan(torch.as_tensor(batch.terminal_payoff, device=self._device))
            if torch.any(terminal_mask):
                payoff = torch.as_tensor(batch.terminal_payoff, dtype=torch.float32, device=self._device)
                ev_p0[terminal_mask] = payoff[terminal_mask]
                ev_p1[terminal_mask] = -payoff[terminal_mask]
            if batch.node_states is not None:
                for index, node_state in enumerate(batch.node_states):
                    if terminal_mask[index]:
                        continue
                    if node_state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
                        continue
                    payouts = compute_payouts(node_state)
                    payout_p0 = next(
                        (payout.amount for payout in payouts if payout.player == 0),
                        0,
                    )
                    payout_p1 = next(
                        (payout.amount for payout in payouts if payout.player == 1),
                        0,
                    )
                    ev_p0[index] = float(payout_p0)
                    ev_p1[index] = float(payout_p1)
            return LeafValueBatch(
                values=np.stack(
                    (
                        ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
                        ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
                    ),
                    axis=1,
                ),
                ev_player0=ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
                ev_player1=ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
                ev_player2=(-(ev_p0 + ev_p1)).detach().to("cpu", dtype=torch.float32).numpy(),
            )
        except Exception:
            return self._cpu_fallback.evaluate(batch)

    def evaluate_tensors(self, tensors: dict[str, torch.Tensor]) -> LeafValueBatch:
        if self._device.type != "cuda":
            raise RuntimeError("GPU tensors require a CUDA device")
        ev_p0 = torch.zeros(tensors["street"].shape[0], dtype=torch.float32, device=self._device)
        ev_p1 = torch.zeros_like(ev_p0)
        terminal_payoff = tensors.get("terminal_payoff")
        if terminal_payoff is not None:
            terminal_mask = ~torch.isnan(terminal_payoff)
            if torch.any(terminal_mask):
                ev_p0[terminal_mask] = terminal_payoff[terminal_mask]
                ev_p1[terminal_mask] = -terminal_payoff[terminal_mask]
        return LeafValueBatch(
            values=np.stack(
                (
                    ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
                    ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
                ),
                axis=1,
            ),
            ev_player0=ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
            ev_player1=ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
            ev_player2=(-(ev_p0 + ev_p1)).detach().to("cpu", dtype=torch.float32).numpy(),
        )

    def evaluate_many(self, batches: tuple[LeafFeatureBatch, ...]) -> tuple[LeafValueBatch, ...]:
        return tuple(self.evaluate(batch) for batch in batches)
