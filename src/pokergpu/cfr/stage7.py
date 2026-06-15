from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DenseCfrState:
    regret_sums: tuple[tuple[float, ...], ...]
    strategy_sums: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.regret_sums) != len(self.strategy_sums):
            raise ValueError("regret and strategy tables must have the same size")
        for regrets, strategy_sums in zip(self.regret_sums, self.strategy_sums, strict=True):
            if len(regrets) != len(strategy_sums):
                raise ValueError("regret and strategy rows must have the same width")


def regret_matching(regrets: tuple[float, ...]) -> tuple[float, ...]:
    if not regrets:
        raise ValueError("regret vector cannot be empty")

    positive = tuple(max(0.0, regret) for regret in regrets)
    total = sum(positive)
    if total <= 0.0:
        return tuple(1.0 / len(regrets) for _ in regrets)
    return tuple(value / total for value in positive)


def update_regret(
    regret_sums: tuple[float, ...],
    action_values: tuple[float, ...],
    node_value: float,
) -> tuple[float, ...]:
    if len(regret_sums) != len(action_values):
        raise ValueError("regrets and action values must have the same length")
    return tuple(
        old_regret + (action_value - node_value)
        for old_regret, action_value in zip(regret_sums, action_values, strict=True)
    )


def update_average_strategy(
    strategy_sums: tuple[float, ...],
    strategy: tuple[float, ...],
    reach_weight: float,
) -> tuple[float, ...]:
    if len(strategy_sums) != len(strategy):
        raise ValueError("strategy sums and strategy must have the same length")
    return tuple(
        old_sum + reach_weight * action_prob
        for old_sum, action_prob in zip(strategy_sums, strategy, strict=True)
    )
