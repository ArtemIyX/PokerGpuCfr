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
    DCFR = "dcfr"


@dataclass(frozen=True, slots=True)
class CFRIterationResult:
    strategies: tuple[NDArray[np.float32], ...]
    infoset_values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class DCFRConfig:
    positive_regret_alpha: float = 1.5
    negative_regret_beta: float = 0.0
    strategy_gamma: float = 2.0


def run_cfr_iteration(
    store: InfosetStore,
    action_utilities: Sequence[NDArray[np.float32] | Sequence[float]],
    strategy_weight: float = 1.0,
    active_infosets: Sequence[int] | None = None,
    variant: CFRVariant = CFRVariant.VANILLA,
    iteration: int | None = None,
    dcfr_config: DCFRConfig | None = None,
) -> CFRIterationResult:
    if len(action_utilities) != store.layout.infoset_count:
        raise ValueError("action utilities must match infoset count")
    if strategy_weight < 0.0:
        raise ValueError("strategy weight must be non-negative")
    if variant is CFRVariant.DCFR and (iteration is None or iteration <= 0):
        raise ValueError("dcfr requires a positive iteration number")
    infoset_indices = (
        tuple(range(store.layout.infoset_count))
        if active_infosets is None
        else tuple(active_infosets)
    )
    config = dcfr_config or DCFRConfig()

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

        if variant is CFRVariant.DCFR:
            _apply_dcfr_discount(
                regrets=regrets,
                strategy_sums=strategy_sums,
                iteration=iteration,
                config=config,
            )

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


def _apply_dcfr_discount(
    regrets: NDArray[np.float32],
    strategy_sums: NDArray[np.float32],
    iteration: int | None,
    config: DCFRConfig,
) -> None:
    assert iteration is not None
    positive_scale = np.float32(
        iteration**config.positive_regret_alpha
        / (iteration**config.positive_regret_alpha + 1.0)
    )
    negative_scale = np.float32(
        iteration**config.negative_regret_beta
        / (iteration**config.negative_regret_beta + 1.0)
    )
    strategy_scale = np.float32(
        (iteration / (iteration + 1.0)) ** config.strategy_gamma
    )

    positive_mask = regrets > 0.0
    negative_mask = regrets < 0.0
    regrets[positive_mask] *= positive_scale
    regrets[negative_mask] *= negative_scale
    strategy_sums *= strategy_scale
