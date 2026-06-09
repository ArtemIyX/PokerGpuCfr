from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .betting import BettingRoundState, PlayerIndex
from .board import Board, Street
from .cards import Card


class HandPhase(StrEnum):
    IN_PROGRESS = "in_progress"
    SHOWDOWN = "showdown"
    TERMINAL = "terminal"


@dataclass(slots=True, frozen=True)
class PlayerState:
    player: PlayerIndex
    hole_cards: tuple[Card, Card] | None = None
    folded: bool = False
    all_in: bool = False

    def __post_init__(self) -> None:
        if self.player < 0:
            raise ValueError("player index must be non-negative")
        if self.folded and self.all_in:
            raise ValueError("player cannot be folded and all-in")
        if self.hole_cards is not None and len(set(self.hole_cards)) != 2:
            raise ValueError("hole cards must be two unique cards")


@dataclass(slots=True, frozen=True)
class GameState:
    board: Board
    players: tuple[PlayerState, ...]
    betting_round: BettingRoundState
    phase: HandPhase = HandPhase.IN_PROGRESS
    dealer: PlayerIndex = PlayerIndex(0)

    def __post_init__(self) -> None:
        if not self.players:
            raise ValueError("game state requires at least one player")
        if self.dealer < 0:
            raise ValueError("dealer index must be non-negative")

        player_ids = {player.player for player in self.players}
        if len(player_ids) != len(self.players):
            raise ValueError("player state entries must be unique")

        stack_ids = {stack.player for stack in self.betting_round.stacks}
        if player_ids != stack_ids:
            raise ValueError("player states must match betting round players")
        if self.dealer not in player_ids:
            raise ValueError("dealer must reference a valid player")

        seen_cards: set[Card] = set(self.board.cards)
        for player in self.players:
            if player.hole_cards is None:
                continue
            for card in player.hole_cards:
                if card in seen_cards:
                    raise ValueError("duplicate card across board and hole cards")
                seen_cards.add(card)

        if self.board.street is not self.current_street:
            raise ValueError("board street and current street must align")

    @property
    def current_street(self) -> Street:
        return self.board.street

    @property
    def active_players(self) -> tuple[PlayerState, ...]:
        return tuple(player for player in self.players if not player.folded)

    @property
    def showdown_eligible_players(self) -> tuple[PlayerState, ...]:
        return tuple(
            player for player in self.players if not player.folded and player.hole_cards
        )

    @property
    def folded_players(self) -> tuple[PlayerState, ...]:
        return tuple(player for player in self.players if player.folded)

    @property
    def player_count(self) -> int:
        return len(self.players)

    def player_state(self, player: PlayerIndex) -> PlayerState:
        match = next((entry for entry in self.players if entry.player == player), None)
        if match is None:
            raise ValueError("unknown player")
        return match
