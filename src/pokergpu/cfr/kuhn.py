from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np
from numpy.typing import NDArray

from .infosets import InfosetLayout, InfosetStore


class KuhnCard(IntEnum):
    JACK = 0
    QUEEN = 1
    KING = 2


class KuhnAction(StrEnum):
    CHECK = "check"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"


@dataclass(frozen=True, slots=True)
class KuhnState:
    cards: tuple[KuhnCard, KuhnCard]
    history: tuple[KuhnAction, ...] = ()

    @property
    def player_to_act(self) -> int:
        if self.is_terminal:
            raise ValueError("terminal states do not have a player to act")
        return len(self.history) % 2

    @property
    def is_terminal(self) -> bool:
        return self.history in {
            (KuhnAction.CHECK, KuhnAction.CHECK),
            (KuhnAction.BET, KuhnAction.CALL),
            (KuhnAction.BET, KuhnAction.FOLD),
            (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.CALL),
            (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.FOLD),
        }

    @property
    def pot_size(self) -> int:
        if self.history in {
            (KuhnAction.BET, KuhnAction.CALL),
            (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.CALL),
        }:
            return 4
        return 2

    def legal_actions(self) -> tuple[KuhnAction, KuhnAction]:
        if self.is_terminal:
            raise ValueError("terminal states have no legal actions")
        if self.history == ():
            return (KuhnAction.CHECK, KuhnAction.BET)
        if self.history == (KuhnAction.CHECK,):
            return (KuhnAction.CHECK, KuhnAction.BET)
        if self.history in {
            (KuhnAction.BET,),
            (KuhnAction.CHECK, KuhnAction.BET),
        }:
            return (KuhnAction.CALL, KuhnAction.FOLD)
        raise ValueError(f"unrecognized Kuhn history: {self.history!r}")

    def apply_action(self, action: KuhnAction) -> KuhnState:
        if action not in self.legal_actions():
            raise ValueError(
                f"illegal Kuhn action {action} for history {self.history!r}"
            )
        return KuhnState(cards=self.cards, history=(*self.history, action))

    def payoff(self, player: int) -> float:
        if not self.is_terminal:
            raise ValueError("payoff is only defined for terminal states")

        if self.history in {
            (KuhnAction.BET, KuhnAction.FOLD),
            (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.FOLD),
        }:
            winner = 1 - ((len(self.history) - 1) % 2)
            return 1.0 if player == winner else -1.0

        winner = 0 if self.cards[0] > self.cards[1] else 1
        showdown_payoff = 2.0 if self.pot_size == 4 else 1.0
        return showdown_payoff if player == winner else -showdown_payoff


@dataclass(frozen=True, slots=True)
class KuhnInfoset:
    player: int
    card: KuhnCard
    history: tuple[KuhnAction, ...]


_INFOSETS: tuple[KuhnInfoset, ...] = tuple(
    KuhnInfoset(player=player, card=card, history=history)
    for player, history in (
        (0, ()),
        (1, (KuhnAction.CHECK,)),
        (1, (KuhnAction.BET,)),
        (0, (KuhnAction.CHECK, KuhnAction.BET)),
    )
    for card in (KuhnCard.JACK, KuhnCard.QUEEN, KuhnCard.KING)
)
_INFOSET_INDEX: dict[KuhnInfoset, int] = {
    infoset: index for index, infoset in enumerate(_INFOSETS)
}
_DEALS: tuple[tuple[KuhnCard, KuhnCard], ...] = tuple(
    (first, second)
    for first in (KuhnCard.JACK, KuhnCard.QUEEN, KuhnCard.KING)
    for second in (KuhnCard.JACK, KuhnCard.QUEEN, KuhnCard.KING)
    if first != second
)


def kuhn_infosets() -> tuple[KuhnInfoset, ...]:
    return _INFOSETS


def kuhn_infoset_layout() -> InfosetLayout:
    return InfosetLayout.from_action_counts([2] * len(_INFOSETS))


def new_kuhn_infoset_store() -> InfosetStore:
    return InfosetStore.zeros(kuhn_infoset_layout())


def expected_action_utilities(
    store: InfosetStore,
    updating_player: int,
) -> tuple[NDArray[np.float32], ...]:
    if updating_player not in {0, 1}:
        raise ValueError("updating player must be 0 or 1")
    if store.layout != kuhn_infoset_layout():
        raise ValueError("store layout must match Kuhn infoset layout")

    utilities = [
        np.zeros(action_count, dtype=np.float32)
        for action_count in store.layout.action_counts
    ]

    for cards in _DEALS:
        _walk_tree(
            store=store,
            state=KuhnState(cards=cards),
            updating_player=updating_player,
            reach_updating=1.0,
            reach_opponent=1.0 / len(_DEALS),
            utilities=utilities,
        )

    return tuple(utilities)


def _walk_tree(
    store: InfosetStore,
    state: KuhnState,
    updating_player: int,
    reach_updating: float,
    reach_opponent: float,
    utilities: list[NDArray[np.float32]],
) -> float:
    if state.is_terminal:
        return state.payoff(updating_player)

    infoset_index = _INFOSET_INDEX[
        KuhnInfoset(
            player=state.player_to_act,
            card=state.cards[state.player_to_act],
            history=state.history,
        )
    ]
    legal_actions = state.legal_actions()
    strategy = store.current_strategy(infoset_index)

    action_values = np.zeros(len(legal_actions), dtype=np.float32)
    node_value = 0.0
    for action_index, action in enumerate(legal_actions):
        if state.player_to_act == updating_player:
            child_value = _walk_tree(
                store=store,
                state=state.apply_action(action),
                updating_player=updating_player,
                reach_updating=reach_updating * float(strategy[action_index]),
                reach_opponent=reach_opponent,
                utilities=utilities,
            )
        else:
            child_value = _walk_tree(
                store=store,
                state=state.apply_action(action),
                updating_player=updating_player,
                reach_updating=reach_updating,
                reach_opponent=reach_opponent * float(strategy[action_index]),
                utilities=utilities,
            )
        action_values[action_index] = np.float32(child_value)
        node_value += float(strategy[action_index]) * child_value

    if state.player_to_act == updating_player:
        utilities[infoset_index] += action_values * np.float32(reach_opponent)

    return node_value
