from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np
from numpy.typing import NDArray

from .infosets import InfosetLayout, InfosetStore
from .iteration import CFRVariant, DCFRConfig, run_cfr_iteration


class LeducRank(IntEnum):
    JACK = 0
    QUEEN = 1
    KING = 2


@dataclass(frozen=True, slots=True, order=True)
class LeducCard:
    rank: LeducRank
    suit: int


class LeducAction(StrEnum):
    CHECK = "check"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"


_DECK: tuple[LeducCard, ...] = tuple(
    LeducCard(rank=rank, suit=suit)
    for rank in (LeducRank.JACK, LeducRank.QUEEN, LeducRank.KING)
    for suit in (0, 1)
)


@dataclass(frozen=True, slots=True)
class LeducRoundState:
    history: tuple[LeducAction, ...]
    bet_open: bool


_INITIAL_ROUND = LeducRoundState(history=(), bet_open=False)


@dataclass(frozen=True, slots=True)
class LeducState:
    cards: tuple[LeducCard, LeducCard]
    public_card: LeducCard | None = None
    round_index: int = 0
    round_state: LeducRoundState = _INITIAL_ROUND
    contributions: tuple[int, int] = (1, 1)
    active_players: tuple[bool, bool] = (True, True)

    @property
    def is_terminal(self) -> bool:
        return not all(self.active_players) or (
            self.round_index == 1
            and self.round_state.history in {
                (LeducAction.CHECK, LeducAction.CHECK),
                (LeducAction.BET, LeducAction.CALL),
                (LeducAction.BET, LeducAction.FOLD),
                (LeducAction.CHECK, LeducAction.BET, LeducAction.CALL),
                (LeducAction.CHECK, LeducAction.BET, LeducAction.FOLD),
            }
        )

    @property
    def needs_public_chance(self) -> bool:
        return (
            self.public_card is None
            and self.round_index == 0
            and self._round_complete()
            and all(self.active_players)
        )

    @property
    def player_to_act(self) -> int:
        if self.is_terminal or self.needs_public_chance:
            raise ValueError("state has no active player")
        return len(self.round_state.history) % 2

    def legal_actions(self) -> tuple[LeducAction, LeducAction]:
        if self.is_terminal or self.needs_public_chance:
            raise ValueError("state has no legal actions")
        history = self.round_state.history
        if history == ():
            return (LeducAction.CHECK, LeducAction.BET)
        if history == (LeducAction.CHECK,):
            return (LeducAction.CHECK, LeducAction.BET)
        if history in {
            (LeducAction.BET,),
            (LeducAction.CHECK, LeducAction.BET),
        }:
            return (LeducAction.CALL, LeducAction.FOLD)
        raise ValueError(f"unrecognized Leduc round history: {history!r}")

    def apply_action(self, action: LeducAction) -> LeducState:
        if action not in self.legal_actions():
            raise ValueError(
                "illegal Leduc action "
                f"{action} for history {self.round_state.history!r}"
            )

        player = self.player_to_act
        next_history = (*self.round_state.history, action)
        contributions = list(self.contributions)
        active_players = list(self.active_players)

        if action is LeducAction.BET:
            contributions[player] += self._bet_size()
        elif action is LeducAction.CALL:
            contributions[player] += self._bet_size()
        elif action is LeducAction.FOLD:
            active_players[player] = False

        next_state = LeducState(
            cards=self.cards,
            public_card=self.public_card,
            round_index=self.round_index,
            round_state=LeducRoundState(
                history=next_history,
                bet_open=action is LeducAction.BET
                or self.round_state.bet_open
                or action is LeducAction.CALL,
            ),
            contributions=(contributions[0], contributions[1]),
            active_players=(active_players[0], active_players[1]),
        )
        if next_state.is_terminal or next_state.needs_public_chance:
            return next_state
        if next_state._round_complete():
            return next_state._advance_round()
        return next_state

    def reveal_public_card(self, card: LeducCard) -> LeducState:
        if not self.needs_public_chance:
            raise ValueError("state does not need public chance")
        if card in self.cards:
            raise ValueError("public card must differ from private cards")
        return LeducState(
            cards=self.cards,
            public_card=card,
            round_index=1,
            round_state=_INITIAL_ROUND,
            contributions=self.contributions,
            active_players=self.active_players,
        )

    def payoff(self, player: int) -> float:
        if not self.is_terminal:
            raise ValueError("payoff is only defined for terminal states")
        pot = sum(self.contributions)
        if not self.active_players[0]:
            return float(self.contributions[player] * -1) if player == 0 else float(
                pot - self.contributions[player]
            )
        if not self.active_players[1]:
            return float(pot - self.contributions[player]) if player == 0 else float(
                -self.contributions[player]
            )

        winner = self._showdown_winner()
        if winner is None:
            return 0.0
        return (
            float(pot - self.contributions[player])
            if player == winner
            else float(-self.contributions[player])
        )

    def _bet_size(self) -> int:
        return 2 if self.round_index == 0 else 4

    def _round_complete(self) -> bool:
        return self.round_state.history in {
            (LeducAction.CHECK, LeducAction.CHECK),
            (LeducAction.BET, LeducAction.CALL),
            (LeducAction.BET, LeducAction.FOLD),
            (LeducAction.CHECK, LeducAction.BET, LeducAction.CALL),
            (LeducAction.CHECK, LeducAction.BET, LeducAction.FOLD),
        }

    def _advance_round(self) -> LeducState:
        if self.public_card is None:
            return self
        return LeducState(
            cards=self.cards,
            public_card=self.public_card,
            round_index=1,
            round_state=_INITIAL_ROUND,
            contributions=self.contributions,
            active_players=self.active_players,
        )

    def _showdown_winner(self) -> int | None:
        assert self.public_card is not None
        player0_pair = self.cards[0].rank == self.public_card.rank
        player1_pair = self.cards[1].rank == self.public_card.rank
        if player0_pair and not player1_pair:
            return 0
        if player1_pair and not player0_pair:
            return 1
        if self.cards[0].rank > self.cards[1].rank:
            return 0
        if self.cards[1].rank > self.cards[0].rank:
            return 1
        return None


@dataclass(frozen=True, slots=True)
class LeducInfoset:
    player: int
    private_rank: LeducRank
    public_rank: LeducRank | None
    round_index: int
    history: tuple[LeducAction, ...]


_PRIVATE_DEALS: tuple[tuple[LeducCard, LeducCard], ...] = tuple(
    (first, second)
    for first in _DECK
    for second in _DECK
    if first != second
)


def _remaining_public_cards(state: LeducState) -> tuple[LeducCard, ...]:
    return tuple(card for card in _DECK if card not in state.cards)


def _build_infosets() -> tuple[tuple[LeducInfoset, ...], dict[LeducInfoset, int]]:
    found: set[LeducInfoset] = set()

    def walk(state: LeducState) -> None:
        if state.is_terminal:
            return
        if state.needs_public_chance:
            for public_card in _remaining_public_cards(state):
                walk(state.reveal_public_card(public_card))
            return
        found.add(
            LeducInfoset(
                player=state.player_to_act,
                private_rank=state.cards[state.player_to_act].rank,
                public_rank=(
                    None if state.public_card is None else state.public_card.rank
                ),
                round_index=state.round_index,
                history=state.round_state.history,
            )
        )
        for action in state.legal_actions():
            walk(state.apply_action(action))

    for cards in _PRIVATE_DEALS:
        walk(LeducState(cards=cards))

    infosets = tuple(sorted(found, key=_infoset_sort_key))
    return infosets, {infoset: index for index, infoset in enumerate(infosets)}


def _infoset_sort_key(
    infoset: LeducInfoset,
) -> tuple[int, int, int, int, tuple[str, ...]]:
    public_rank = -1 if infoset.public_rank is None else int(infoset.public_rank)
    return (
        infoset.player,
        infoset.round_index,
        int(infoset.private_rank),
        public_rank,
        tuple(action.value for action in infoset.history),
    )


_INFOSETS, _INFOSET_INDEX = _build_infosets()


def leduc_infosets() -> tuple[LeducInfoset, ...]:
    return _INFOSETS


def leduc_infoset_indices_for_player(player: int) -> tuple[int, ...]:
    if player not in {0, 1}:
        raise ValueError("player must be 0 or 1")
    return tuple(
        index for index, infoset in enumerate(_INFOSETS) if infoset.player == player
    )


def leduc_infoset_layout() -> InfosetLayout:
    return InfosetLayout.from_action_counts([2] * len(_INFOSETS))


def new_leduc_infoset_store() -> InfosetStore:
    return InfosetStore.zeros(leduc_infoset_layout())


def expected_action_utilities_leduc(
    store: InfosetStore,
    updating_player: int,
) -> tuple[NDArray[np.float32], ...]:
    if updating_player not in {0, 1}:
        raise ValueError("updating player must be 0 or 1")
    if store.layout != leduc_infoset_layout():
        raise ValueError("store layout must match Leduc infoset layout")

    utilities = [
        np.zeros(action_count, dtype=np.float32)
        for action_count in store.layout.action_counts
    ]
    chance_weight = 1.0 / len(_PRIVATE_DEALS)
    for cards in _PRIVATE_DEALS:
        _walk_tree(
            store=store,
            state=LeducState(cards=cards),
            updating_player=updating_player,
            reach_opponent=chance_weight,
            utilities=utilities,
        )
    return tuple(utilities)


def train_leduc_cfr(
    iterations: int,
    variant: CFRVariant = CFRVariant.VANILLA,
    dcfr_config: DCFRConfig | None = None,
) -> InfosetStore:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    store = new_leduc_infoset_store()
    for iteration_index in range(1, iterations + 1):
        for player in (0, 1):
            run_cfr_iteration(
                store=store,
                action_utilities=expected_action_utilities_leduc(store, player),
                active_infosets=leduc_infoset_indices_for_player(player),
                variant=variant,
                iteration=iteration_index,
                dcfr_config=dcfr_config,
            )
    return store


def average_strategy_profile_leduc(
    store: InfosetStore,
) -> dict[LeducInfoset, NDArray[np.float32]]:
    if store.layout != leduc_infoset_layout():
        raise ValueError("store layout must match Leduc infoset layout")
    return {
        infoset: store.average_strategy(index)
        for index, infoset in enumerate(_INFOSETS)
    }


def expected_game_value_for_average_strategy_leduc(store: InfosetStore) -> float:
    return expected_game_value_leduc(average_strategy_profile_leduc(store), player=0)


def average_strategy_root_bet_probability_leduc(
    store: InfosetStore,
    rank: LeducRank,
) -> float:
    infoset = LeducInfoset(
        player=0,
        private_rank=rank,
        public_rank=None,
        round_index=0,
        history=(),
    )
    return float(average_strategy_profile_leduc(store)[infoset][1])


def expected_game_value_leduc(
    strategy_profile: dict[LeducInfoset, NDArray[np.float32]],
    player: int,
) -> float:
    total = 0.0
    chance_weight = 1.0 / len(_PRIVATE_DEALS)
    for cards in _PRIVATE_DEALS:
        total += chance_weight * _walk_average_strategy_tree(
            strategy_profile=strategy_profile,
            state=LeducState(cards=cards),
            player=player,
        )
    return total


def _infoset_for_state(state: LeducState) -> LeducInfoset:
    return LeducInfoset(
        player=state.player_to_act,
        private_rank=state.cards[state.player_to_act].rank,
        public_rank=None if state.public_card is None else state.public_card.rank,
        round_index=state.round_index,
        history=state.round_state.history,
    )


def _walk_tree(
    store: InfosetStore,
    state: LeducState,
    updating_player: int,
    reach_opponent: float,
    utilities: list[NDArray[np.float32]],
) -> float:
    if state.is_terminal:
        return state.payoff(updating_player)
    if state.needs_public_chance:
        chance_weight = 1.0 / len(_remaining_public_cards(state))
        total = 0.0
        for public_card in _remaining_public_cards(state):
            total += chance_weight * _walk_tree(
                store=store,
                state=state.reveal_public_card(public_card),
                updating_player=updating_player,
                reach_opponent=reach_opponent,
                utilities=utilities,
            )
        return total

    infoset_index = _INFOSET_INDEX[_infoset_for_state(state)]
    strategy = store.current_strategy(infoset_index)
    action_values = np.zeros(2, dtype=np.float32)
    node_value = 0.0
    for action_index, action in enumerate(state.legal_actions()):
        next_reach_opponent = (
            reach_opponent * float(strategy[action_index])
            if state.player_to_act != updating_player
            else reach_opponent
        )
        child_value = _walk_tree(
            store=store,
            state=state.apply_action(action),
            updating_player=updating_player,
            reach_opponent=next_reach_opponent,
            utilities=utilities,
        )
        action_values[action_index] = np.float32(child_value)
        node_value += float(strategy[action_index]) * child_value

    if state.player_to_act == updating_player:
        utilities[infoset_index] += action_values * np.float32(reach_opponent)
    return node_value


def _walk_average_strategy_tree(
    strategy_profile: dict[LeducInfoset, NDArray[np.float32]],
    state: LeducState,
    player: int,
) -> float:
    if state.is_terminal:
        return state.payoff(player)
    if state.needs_public_chance:
        chance_weight = 1.0 / len(_remaining_public_cards(state))
        return sum(
            chance_weight
            * _walk_average_strategy_tree(
                strategy_profile=strategy_profile,
                state=state.reveal_public_card(public_card),
                player=player,
            )
            for public_card in _remaining_public_cards(state)
        )
    strategy = strategy_profile[_infoset_for_state(state)]
    return sum(
        float(strategy[action_index])
        * _walk_average_strategy_tree(
            strategy_profile=strategy_profile,
            state=state.apply_action(action),
            player=player,
        )
        for action_index, action in enumerate(state.legal_actions())
    )
