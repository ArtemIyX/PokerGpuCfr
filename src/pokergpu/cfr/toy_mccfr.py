from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .infosets import InfosetLayout, InfosetStore


@dataclass(frozen=True, slots=True)
class ToyMCCFRResult:
    store: InfosetStore
    expected_value_p0: float


def new_toy_store() -> InfosetStore:
    return InfosetStore.zeros(InfosetLayout.from_action_counts([2, 2]))


def toy_expected_value() -> float:
    return 0.0


def train_toy_mccfr(iterations: int, seed: int = 0) -> ToyMCCFRResult:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = np.random.default_rng(seed)
    store = new_toy_store()
    for _ in range(iterations):
        _sample_iteration(store, rng)
    return ToyMCCFRResult(store=store, expected_value_p0=_evaluate_average(store))


def _sample_iteration(store: InfosetStore, rng: np.random.Generator) -> None:
    _traverse(store, node=0, reach_p0=1.0, reach_p1=1.0, rng=rng)


def _traverse(
    store: InfosetStore,
    node: int,
    reach_p0: float,
    reach_p1: float,
    rng: np.random.Generator,
) -> float:
    if node == 3:
        return 1.0
    if node == 4:
        return -1.0
    if node == 5:
        return -1.0
    if node == 6:
        return 1.0

    infoset_index = 0 if node == 0 else 1
    strategy = store.current_strategy(infoset_index)
    if node == 0:
        action_values = np.array(
            [
                _traverse(store, 1, reach_p0, reach_p1, rng),
                _traverse(store, 2, reach_p0, reach_p1, rng),
            ],
            dtype=np.float32,
        )
        node_value = float(np.dot(strategy, action_values))
        store.regrets_for_infoset(0)[:] += action_values - np.float32(node_value)
        store.strategy_sums_for_infoset(0)[:] += strategy
        return node_value

    action_values = np.array(
        [
            _traverse(store, 3 if node == 1 else 5, reach_p0, reach_p1, rng),
            _traverse(store, 4 if node == 1 else 6, reach_p0, reach_p1, rng),
        ],
        dtype=np.float32,
    )
    node_value = float(np.dot(strategy, action_values))
    store.regrets_for_infoset(1)[:] += action_values - np.float32(node_value)
    store.strategy_sums_for_infoset(1)[:] += strategy
    return node_value


def _evaluate_average(store: InfosetStore) -> float:
    root = store.average_strategy(0)
    child = store.average_strategy(1)
    p0_left = float(np.dot(child, np.array([1.0, -1.0], dtype=np.float32)))
    p0_right = float(np.dot(child, np.array([-1.0, 1.0], dtype=np.float32)))
    return float(np.dot(root, np.array([p0_left, p0_right], dtype=np.float32)))
