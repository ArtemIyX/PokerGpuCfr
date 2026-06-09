from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .infosets import InfosetStore


class CFRVariant(StrEnum):
    VANILLA = "vanilla"
    CFR_PLUS = "cfr_plus"


@dataclass(frozen=True, slots=True)
class CFRIterationResult:
    strategies: tuple[NDArray[np.float32], ...]
    infoset_values: NDArray[np.float32]


def run_cfr_iteration(
    store: InfosetStore,
    action_utilities: Sequence[NDArray[np.float32] | Sequence[float]],
    strategy_weight: float = 1.0,
    active_infosets: Sequence[int] | None = None,
    variant: CFRVariant = CFRVariant.VANILLA,
) -> CFRIterationResult:
    if len(action_utilities) != store.layout.infoset_count:
        raise ValueError("action utilities must match infoset count")
    if strategy_weight < 0.0:
        raise ValueError("strategy weight must be non-negative")
    infoset_indices = (
        tuple(range(store.layout.infoset_count))
        if active_infosets is None
        else tuple(active_infosets)
    )

    strategies: list[NDArray[np.float32]] = []
    infoset_values = np.zeros(store.layout.infoset_count, dtype=np.float32)
    strategy_scale = np.float32(strategy_weight)

    for infoset_index in infoset_indices:
        if infoset_index < 0 or infoset_index >= store.layout.infoset_count:
            raise IndexError(f"infoset index out of range: {infoset_index}")
        utility_values_like = action_utilities[infoset_index]
        utility_values = np.asarray(utility_values_like, dtype=np.float32)
        if utility_values.ndim != 1:
            raise ValueError("action utility arrays must be one-dimensional")

        regrets = store.regrets_for_infoset(infoset_index)
        strategy_sums = store.strategy_sums_for_infoset(infoset_index)
        if utility_values.shape[0] != regrets.shape[0]:
            raise ValueError("action utility length must match infoset action count")

        strategy = store.current_strategy(infoset_index)
        infoset_value = np.float32(
            np.sum(strategy * utility_values, dtype=np.float64)
        )
        regrets += utility_values - infoset_value
        if variant is CFRVariant.CFR_PLUS:
            np.maximum(regrets, np.float32(0.0), out=regrets)
        strategy_sums += strategy * strategy_scale

        strategies.append(strategy)
        infoset_values[infoset_index] = infoset_value

    return CFRIterationResult(
        strategies=tuple(strategies),
        infoset_values=infoset_values,
    )
