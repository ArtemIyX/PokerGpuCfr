from __future__ import annotations

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
            _ = build_gpu_leaf_tensors(batch, self._device)
            ev_p0 = torch.zeros(batch.size, dtype=torch.float32, device=self._device)
            ev_p1 = torch.zeros(batch.size, dtype=torch.float32, device=self._device)
            for index in range(batch.size):
                payoff = float(batch.terminal_payoff[index])
                if not torch.isnan(torch.tensor(payoff)):
                    ev_p0[index] = payoff
                    ev_p1[index] = -payoff
                    continue
                node_state = None if batch.node_states is None else batch.node_states[index]
                if node_state is None:
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
                ev_player0=ev_p0.detach().to("cpu", dtype=torch.float32).numpy(),
                ev_player1=ev_p1.detach().to("cpu", dtype=torch.float32).numpy(),
                ev_player2=(-(ev_p0 + ev_p1)).detach().to("cpu", dtype=torch.float32).numpy(),
            )
        except Exception:
            return self._cpu_fallback.evaluate(batch)
