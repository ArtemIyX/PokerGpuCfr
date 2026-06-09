from __future__ import annotations

from .state import GameState


def public_state_signature(state: GameState) -> str:
    player_parts = []
    for player in sorted(state.players, key=lambda entry: entry.player):
        bet = next(
            entry for entry in state.betting_round.bets if entry.player == player.player
        )
        stack = next(
            entry
            for entry in state.betting_round.stacks
            if entry.player == player.player
        )
        player_parts.append(
            ":".join(
                (
                    str(player.player),
                    f"folded={int(player.folded)}",
                    f"all_in={int(player.all_in)}",
                    f"stack={stack.stack}",
                    f"committed={bet.committed}",
                )
            )
        )

    return "|".join(
        (
            f"phase={state.phase.value}",
            f"street={state.current_street.value}",
            f"board={state.board}",
            f"dealer={state.dealer}",
            f"to_act={state.betting_round.to_act}",
            f"pot={state.betting_round.pot.amount}",
            f"sb={state.betting_round.blinds.small_blind}",
            f"bb={state.betting_round.blinds.big_blind}",
            f"ante={state.betting_round.blinds.ante}",
            ";".join(player_parts),
        )
    )
