from __future__ import annotations

import numpy as np

from .interface import LeafEvaluator
from .types import LeafFeatureBatch, LeafValueBatch


class CpuStubLeafEvaluator(LeafEvaluator):
    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        payoff = np.asarray(batch.terminal_payoff, dtype=np.float32)
        terminal_mask = ~np.isnan(payoff)

        ev_p0 = np.where(terminal_mask, payoff, 0.0).astype(np.float32, copy=False)
        ev_p1 = np.where(terminal_mask, -payoff, 0.0).astype(np.float32, copy=False)
        ev_p2 = np.zeros_like(ev_p0, dtype=np.float32)

        return LeafValueBatch(
            values=np.stack((ev_p0, ev_p1, ev_p2), axis=1),
            ev_player0=ev_p0,
            ev_player1=ev_p1,
            ev_player2=ev_p2,
        )
