from __future__ import annotations

import numpy as np

from pokergpu.core.state import HandPhase

from .interface import LeafEvaluator
from .types import LeafFeatureBatch, LeafValueBatch


class CpuStubLeafEvaluator(LeafEvaluator):
    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        from pokergpu.core.payouts import compute_payouts

        payoff = np.asarray(batch.terminal_payoff, dtype=np.float32)
        terminal_mask = ~np.isnan(payoff)
        if not bool(terminal_mask.all()) and batch.node_states is not None:
            node_states = batch.node_states
            for index, is_terminal in enumerate(terminal_mask):
                if is_terminal:
                    continue
                node_state = node_states[index]
                if node_state.phase not in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
                    payoff[index] = _heuristic_terminal_payoff(
                        pot=float(batch.pot[index]),
                        stack_p0=float(batch.stack_p0[index]),
                        stack_p1=float(batch.stack_p1[index]),
                        street=int(batch.street[index]),
                        player_to_act=int(batch.player_to_act[index]),
                        board_size=int(batch.board_size[index]),
                    )
                    terminal_mask[index] = True
                    continue
                payouts = compute_payouts(node_state)
                payoff[index] = np.float32(
                    next((payout.amount for payout in payouts if payout.player == 0), 0)
                    - next((payout.amount for payout in payouts if payout.player != 0), 0)
                )
                terminal_mask[index] = True

        ev_p0 = np.where(terminal_mask, payoff, 0.0).astype(np.float32, copy=False)
        ev_p1 = np.where(terminal_mask, -payoff, 0.0).astype(np.float32, copy=False)
        ev_p2 = np.zeros_like(ev_p0, dtype=np.float32)

        return LeafValueBatch(
            values=np.stack((ev_p0, ev_p1, ev_p2), axis=1),
            ev_player0=ev_p0,
            ev_player1=ev_p1,
            ev_player2=ev_p2,
        )


def _heuristic_terminal_payoff(
    *,
    pot: float,
    stack_p0: float,
    stack_p1: float,
    street: int,
    player_to_act: int,
    board_size: int,
) -> np.float32:
    stack_diff = stack_p0 - stack_p1
    street_bias = (street + 1.0) / 5.0
    board_bias = board_size / 5.0
    to_act_bias = 1.0 if player_to_act == 0 else -1.0
    raw = (
        0.10 * pot * to_act_bias
        + 0.03 * stack_diff
        + 0.02 * pot * (street_bias - 0.5)
        + 0.01 * pot * (board_bias - 0.5)
    )
    return np.float32(raw)
