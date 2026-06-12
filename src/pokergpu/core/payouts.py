from __future__ import annotations

from dataclasses import dataclass

from pokergpu.eval.treys_evaluator import TreysHandEvaluator

from .betting import Chips, PlayerIndex
from .state import GameState
from .terminal import (
    active_players,
    is_hand_complete,
    is_showdown_state,
    is_terminal_state,
)


@dataclass(slots=True, frozen=True)
class Payout:
    player: PlayerIndex
    amount: Chips


def total_pot(state: GameState) -> Chips:
    committed_total = sum(bet.committed for bet in state.betting_round.bets)
    return Chips(state.betting_round.pot.amount + committed_total)


def compute_payouts(
    state: GameState,
    *,
    evaluator: TreysHandEvaluator | None = None,
) -> tuple[Payout, ...]:
    if not is_hand_complete(state):
        raise ValueError("cannot compute payouts for incomplete hand")

    pot = total_pot(state)
    zero_payouts = {player.player: Chips(0) for player in state.players}

    if is_terminal_state(state) and not is_showdown_state(state):
        winners = active_players(state)
        if len(winners) != 1:
            raise ValueError("terminal non-showdown state must have exactly one winner")
        winner_index = winners[0].player
        zero_payouts[winner_index] = pot
        return tuple(
            Payout(player=player.player, amount=zero_payouts[player.player])
            for player in state.players
        )

    evaluator_instance = evaluator or TreysHandEvaluator()
    contenders = tuple(player for player in active_players(state) if player.hole_cards)
    if not contenders:
        raise ValueError("showdown requires at least one player with hole cards")

    scores: dict[PlayerIndex, int] = {}
    for player in contenders:
        hole_cards = player.hole_cards
        if hole_cards is None:
            raise ValueError("showdown contender must have hole cards")
        scores[player.player] = evaluator_instance.evaluate_best_hand(
            hole_cards + state.board.cards
        ).score

    _award_side_pots(state, scores, zero_payouts)

    return tuple(
        Payout(player=player.player, amount=zero_payouts[player.player])
        for player in state.players
    )


def _award_side_pots(
    state: GameState,
    scores: dict[PlayerIndex, int],
    payouts: dict[PlayerIndex, Chips],
) -> None:
    commitments = {
        bet.player: bet.committed
        for bet in state.betting_round.bets
        if bet.committed > 0
    }
    levels = sorted(set(commitments.values()))
    previous_level = 0
    base_pot_awarded = False

    for level in levels:
        contributors = tuple(
            player for player, committed in commitments.items() if committed >= level
        )
        eligible = tuple(
            player
            for player in contributors
            if not state.player_state(player).folded and player in scores
        )
        pot_size = (level - previous_level) * len(contributors)
        if not base_pot_awarded:
            pot_size += state.betting_round.pot.amount
            base_pot_awarded = True
        previous_level = level

        if pot_size <= 0 or not eligible:
            continue

        best_score = min(scores[player] for player in eligible)
        winners = tuple(player for player in eligible if scores[player] == best_score)
        split = pot_size // len(winners)
        remainder = pot_size % len(winners)

        for winner in winners:
            bonus = 1 if remainder > 0 else 0
            payouts[winner] = Chips(payouts[winner] + split + bonus)
            if remainder > 0:
                remainder -= 1
