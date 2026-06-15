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


@dataclass(slots=True, frozen=True)
class ToyCfrState:
    regret_sums: tuple[float, ...]
    strategy_sums: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class ToyCfrResult:
    state: ToyCfrState
    strategy: tuple[float, ...]
    node_value: float
    action_values: tuple[float, ...]
