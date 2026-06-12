from __future__ import annotations

import numpy as np

from .infosets import InfosetStore
from .kuhn import (
    KuhnCard,
    KuhnInfoset,
    KuhnState,
    kuhn_infosets,
    new_kuhn_infoset_store,
)
from .leduc import (
    LeducCard,
    LeducInfoset,
    LeducRank,
    LeducState,
    leduc_infosets,
    new_leduc_infoset_store,
)


def train_kuhn_mccfr(iterations: int, seed: int = 0) -> InfosetStore:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = np.random.default_rng(seed)
    store = new_kuhn_infoset_store()
    infoset_index = {infoset: index for index, infoset in enumerate(kuhn_infosets())}
    deals: tuple[tuple[KuhnCard, KuhnCard], ...] = tuple(
        (KuhnCard(first), KuhnCard(second))
        for first in range(3)
        for second in range(3)
        if first != second
    )
    for _ in range(iterations):
        for updating_player in (0, 1):
            cards = deals[int(rng.integers(len(deals)))]
            _kuhn_traverse(
                store=store,
                state=KuhnState(cards=cards),
                updating_player=updating_player,
                infoset_index=infoset_index,
                rng=rng,
            )
    return store


def train_leduc_mccfr(iterations: int, seed: int = 0) -> InfosetStore:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = np.random.default_rng(seed)
    store = new_leduc_infoset_store()
    infoset_index = {infoset: index for index, infoset in enumerate(leduc_infosets())}
    deck = tuple(
        LeducCard(rank=rank, suit=suit)
        for rank in (LeducRank.JACK, LeducRank.QUEEN, LeducRank.KING)
        for suit in (0, 1)
    )
    private_deals = tuple(
        (first, second) for first in deck for second in deck if first != second
    )
    for _ in range(iterations):
        for updating_player in (0, 1):
            cards = private_deals[int(rng.integers(len(private_deals)))]
            _leduc_traverse(
                store=store,
                state=LeducState(cards=cards),
                updating_player=updating_player,
                infoset_index=infoset_index,
                rng=rng,
            )
    return store


def _kuhn_traverse(
    store: InfosetStore,
    state: KuhnState,
    updating_player: int,
    infoset_index: dict[KuhnInfoset, int],
    rng: np.random.Generator,
) -> float:
    if state.is_terminal:
        return state.payoff(updating_player)

    key = KuhnInfoset(
        player=state.player_to_act,
        card=state.cards[state.player_to_act],
        history=state.history,
    )
    index = infoset_index[key]
    strategy = store.current_strategy(index)
    legal_actions = state.legal_actions()
    action_probs = strategy[: len(legal_actions)]
    action_probs = np.asarray(
        action_probs / np.sum(action_probs, dtype=np.float32),
        dtype=np.float32,
    )
    sampled_action = int(rng.choice(len(legal_actions), p=action_probs))

    action_values = np.zeros(len(legal_actions), dtype=np.float32)
    for action_index, action in enumerate(legal_actions):
        action_values[action_index] = np.float32(
            _kuhn_traverse(
                store,
                state.apply_action(action),
                updating_player,
                infoset_index,
                rng,
            )
        )

    node_value = float(np.dot(strategy[: len(legal_actions)], action_values))
    if state.player_to_act == updating_player:
        regrets = store.regrets_for_infoset(index)
        regrets += action_values - np.float32(node_value)
        strategy_sums = store.strategy_sums_for_infoset(index)
        strategy_sums += strategy[: len(legal_actions)]
    return float(action_values[sampled_action])


def _leduc_traverse(
    store: InfosetStore,
    state: LeducState,
    updating_player: int,
    infoset_index: dict[LeducInfoset, int],
    rng: np.random.Generator,
) -> float:
    if state.is_terminal:
        return state.payoff(updating_player)
    if state.needs_public_chance:
        remaining = tuple(card for card in _leduc_deck() if card not in state.cards)
        public_card = remaining[int(rng.integers(len(remaining)))]
        return _leduc_traverse(
            store,
            state.reveal_public_card(public_card),
            updating_player,
            infoset_index,
            rng,
        )

    key = LeducInfoset(
        player=state.player_to_act,
        private_rank=state.cards[state.player_to_act].rank,
        public_rank=None if state.public_card is None else state.public_card.rank,
        round_index=state.round_index,
        history=state.round_state.history,
    )
    index = infoset_index[key]
    strategy = store.current_strategy(index)
    legal_actions = state.legal_actions()
    action_probs = strategy[: len(legal_actions)]
    action_probs = np.asarray(
        action_probs / np.sum(action_probs, dtype=np.float32),
        dtype=np.float32,
    )
    sampled_action = int(rng.choice(len(legal_actions), p=action_probs))

    action_values = np.zeros(len(legal_actions), dtype=np.float32)
    for action_index, action in enumerate(legal_actions):
        action_values[action_index] = np.float32(
            _leduc_traverse(
                store,
                state.apply_action(action),
                updating_player,
                infoset_index,
                rng,
            )
        )

    node_value = float(np.dot(strategy[: len(legal_actions)], action_values))
    if state.player_to_act == updating_player:
        regrets = store.regrets_for_infoset(index)
        regrets += action_values - np.float32(node_value)
        strategy_sums = store.strategy_sums_for_infoset(index)
        strategy_sums += strategy[: len(legal_actions)]
    return float(action_values[sampled_action])


def _leduc_deck() -> tuple[LeducCard, ...]:
    return tuple(
        LeducCard(rank=rank, suit=suit)
        for rank in (LeducRank.JACK, LeducRank.QUEEN, LeducRank.KING)
        for suit in (0, 1)
    )
