from __future__ import annotations

import numpy as np

from pokergpu.core.state import HandPhase
from .interface import LeafEvaluator
from .types import LeafFeatureBatch, LeafValueBatch


class CpuStubLeafEvaluator(LeafEvaluator):
    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        from pokergpu.core.payouts import compute_payouts

        ev_p0 = np.zeros(batch.size, dtype=np.float32)
        ev_p1 = np.zeros(batch.size, dtype=np.float32)

        for index in range(batch.size):
            node_state = None if batch.node_states is None else batch.node_states[index]
            payoff = batch.terminal_payoff[index]
            if not np.isnan(payoff):
                ev_p0[index] = payoff
                ev_p1[index] = -payoff
                continue
            if node_state is None:
                ev_p0[index] = np.float32(0.0)
                ev_p1[index] = np.float32(0.0)
                continue
            if node_state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
                ev_p0[index] = np.float32(0.0)
                ev_p1[index] = np.float32(0.0)
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
            ev_p0[index] = np.float32(payout_p0)
            ev_p1[index] = np.float32(payout_p1)

        return LeafValueBatch(
            ev_player0=ev_p0,
            ev_player1=ev_p1,
            ev_player2=-(ev_p0 + ev_p1),
        )
