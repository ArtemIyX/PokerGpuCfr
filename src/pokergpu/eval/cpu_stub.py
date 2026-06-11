from __future__ import annotations

import numpy as np

from .interface import LeafEvaluator
from .types import LeafFeatureBatch, LeafValueBatch


class CpuStubLeafEvaluator(LeafEvaluator):
    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        ev_p0 = np.zeros(batch.size, dtype=np.float32)
        ev_p1 = np.zeros(batch.size, dtype=np.float32)

        for index in range(batch.size):
            if batch.is_terminal[index]:
                pot = batch.pot[index]
                bias = np.float32(0.0)
                if batch.player_to_act[index] == 0:
                    bias = np.float32(0.0)
                ev_p0[index] = np.float32(bias)
                ev_p1[index] = np.float32(-bias)
                continue

            pot = batch.pot[index]
            reach_delta = batch.reach_p0[index] - batch.reach_p1[index]
            ev_p0[index] = np.float32(0.5 * pot + reach_delta)
            ev_p1[index] = np.float32(-0.5 * pot - reach_delta)

        return LeafValueBatch(ev_player0=ev_p0, ev_player1=ev_p1)
