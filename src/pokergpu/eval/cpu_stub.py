from __future__ import annotations

import numpy as np

from .interface import LeafEvaluator
from .types import LeafFeatureBatch, LeafValueBatch


class CpuStubLeafEvaluator(LeafEvaluator):
    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        ev_p0 = np.zeros(batch.size, dtype=np.float32)
        ev_p1 = np.zeros(batch.size, dtype=np.float32)

        for index in range(batch.size):
            pot = batch.pot[index]
            board_size = np.float32(batch.board_size[index])
            stack_gap = np.float32(batch.stack_p0[index] - batch.stack_p1[index])
            street = np.float32(batch.street[index])
            reach_delta = batch.reach_p0[index] - batch.reach_p1[index]
            reach_delta += batch.reach_p2[index] * np.float32(0.0)
            terminal_bias = np.float32(0.0 if batch.is_terminal[index] else 1.0)
            value = (
                np.float32(0.0001) * pot
                + np.float32(0.01) * stack_gap
                + np.float32(0.05) * board_size
                + np.float32(0.02) * street
                + np.float32(0.5) * reach_delta
                + np.float32(0.1) * terminal_bias
            )
            ev_p0[index] = value
            ev_p1[index] = -value

        return LeafValueBatch(
            ev_player0=ev_p0,
            ev_player1=ev_p1,
            ev_player2=-(ev_p0 + ev_p1),
        )
