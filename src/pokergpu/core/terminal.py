from __future__ import annotations

from .state import GameState, HandPhase, PlayerState


def active_players(state: GameState) -> tuple[PlayerState, ...]:
    return tuple(player for player in state.players if not player.folded)


def non_all_in_active_players(state: GameState) -> tuple[PlayerState, ...]:
    return tuple(player for player in active_players(state) if not player.all_in)


def is_terminal_state(state: GameState) -> bool:
    if state.phase is HandPhase.TERMINAL:
        return True
    return len(active_players(state)) <= 1


def is_showdown_state(state: GameState) -> bool:
    if state.phase is HandPhase.SHOWDOWN:
        return True
    return len(active_players(state)) > 1 and len(non_all_in_active_players(state)) <= 1


def is_hand_complete(state: GameState) -> bool:
    return is_terminal_state(state) or is_showdown_state(state)
