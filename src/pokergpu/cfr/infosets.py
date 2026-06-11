from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class InfosetLayout:
    action_counts: tuple[int, ...]
    offsets: tuple[int, ...]
    total_actions: int

    def __post_init__(self) -> None:
        if len(self.action_counts) != len(self.offsets):
            raise ValueError("action counts and offsets must have the same length")
        if self.total_actions < 0:
            raise ValueError("total actions must be non-negative")

        expected_offset = 0
        for action_count, offset in zip(self.action_counts, self.offsets, strict=True):
            if action_count <= 0:
                raise ValueError("each infoset must have at least one action")
            if offset != expected_offset:
                raise ValueError("infoset offsets must be contiguous")
            expected_offset += action_count

        if expected_offset != self.total_actions:
            raise ValueError("total actions must match the sum of action counts")

    @classmethod
    def from_action_counts(
        cls,
        action_counts: tuple[int, ...] | list[int],
    ) -> InfosetLayout:
        normalized_counts = tuple(action_counts)
        offsets: list[int] = []
        running_offset = 0
        for action_count in normalized_counts:
            offsets.append(running_offset)
            running_offset += action_count
        return cls(
            action_counts=normalized_counts,
            offsets=tuple(offsets),
            total_actions=running_offset,
        )

    @property
    def infoset_count(self) -> int:
        return len(self.action_counts)

    def action_range(self, infoset_index: int) -> slice:
        if infoset_index < 0 or infoset_index >= self.infoset_count:
            raise IndexError(f"infoset index out of range: {infoset_index}")
        start = self.offsets[infoset_index]
        return slice(start, start + self.action_counts[infoset_index])


def build_infoset_layout(action_counts: tuple[int, ...] | list[int]) -> InfosetLayout:
    return InfosetLayout.from_action_counts(action_counts)


@dataclass(slots=True)
class InfosetStore:
    layout: InfosetLayout
    regrets: NDArray[np.float32]
    strategy_sums: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.regrets.ndim != 1 or self.strategy_sums.ndim != 1:
            raise ValueError("infoset arrays must be one-dimensional")
        if self.regrets.shape[0] != self.layout.total_actions:
            raise ValueError("regret array length must match layout total actions")
        if self.strategy_sums.shape[0] != self.layout.total_actions:
            raise ValueError(
                "strategy sum array length must match layout total actions"
            )

    @classmethod
    def zeros(cls, layout: InfosetLayout) -> InfosetStore:
        return cls(
            layout=layout,
            regrets=np.zeros(layout.total_actions, dtype=np.float32),
            strategy_sums=np.zeros(layout.total_actions, dtype=np.float32),
        )

    def regrets_for_infoset(self, infoset_index: int) -> NDArray[np.float32]:
        return self.regrets[self.layout.action_range(infoset_index)]

    def strategy_sums_for_infoset(self, infoset_index: int) -> NDArray[np.float32]:
        return self.strategy_sums[self.layout.action_range(infoset_index)]

    def current_strategy(self, infoset_index: int) -> NDArray[np.float32]:
        regrets = self.regrets_for_infoset(infoset_index)
        return regret_matching(regrets)

    def average_strategy(self, infoset_index: int) -> NDArray[np.float32]:
        strategy_sums = self.strategy_sums_for_infoset(infoset_index)
        total = float(np.sum(strategy_sums, dtype=np.float64))
        if total <= 0.0:
            action_count = strategy_sums.shape[0]
            value = np.float32(1.0 / action_count)
            return np.full(action_count, value, dtype=np.float32)
        return strategy_sums / np.float32(total)


def regret_matching(regrets: NDArray[np.float32]) -> NDArray[np.float32]:
    if regrets.ndim != 1:
        raise ValueError("regret matching requires a one-dimensional array")
    if regrets.shape[0] == 0:
        raise ValueError("regret matching requires at least one action")

    positive_regrets = np.maximum(regrets, np.float32(0.0))
    total = float(np.sum(positive_regrets, dtype=np.float64))
    if total <= 0.0:
        value = np.float32(1.0 / regrets.shape[0])
        return np.full(regrets.shape[0], value, dtype=np.float32)
    return (positive_regrets / np.float32(total)).astype(np.float32)
