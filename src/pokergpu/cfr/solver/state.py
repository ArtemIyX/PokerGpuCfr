from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DenseCfrState:
    regret_sums: tuple[tuple[float, ...], ...]
    strategy_sums: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.regret_sums and not self.strategy_sums:
            return
        if len(self.regret_sums) != len(self.strategy_sums):
            raise ValueError("regret and strategy tables must have the same size")
        for regrets, strategy_sums in zip(self.regret_sums, self.strategy_sums, strict=True):
            assert regrets, "regret rows cannot be empty"
            if len(regrets) != len(strategy_sums):
                raise ValueError("regret and strategy rows must have the same width")


@dataclass(slots=True, frozen=True)
class SolverState:
    regret_sums: tuple[float, ...]
    strategy_sums: tuple[float, ...]

    def __post_init__(self) -> None:
        assert self.regret_sums or self.strategy_sums, "solver state cannot be empty"
        if len(self.regret_sums) != len(self.strategy_sums):
            raise ValueError("regret and strategy vectors must have the same length")


@dataclass(slots=True, frozen=True)
class SolverIterationResult:
    state: SolverState
    strategy: tuple[float, ...]
    node_value: float
    action_values: tuple[float, ...]

    def __post_init__(self) -> None:
        assert self.strategy, "strategy cannot be empty"
        assert self.action_values, "action values cannot be empty"
