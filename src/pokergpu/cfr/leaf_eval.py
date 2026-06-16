from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


LEAF_EVAL_FEATURE_WIDTH = 161
LEAF_EVAL_OUTPUT_WIDTH = 1


@dataclass(slots=True, frozen=True)
class LeafEvalBatchInput:
    node_ids: tuple[int, ...]
    features: np.ndarray

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("leaf eval features must be a 2D tensor")
        if self.features.dtype != np.float32:
            raise ValueError("leaf eval features must use float32")
        if self.features.shape[1] != LEAF_EVAL_FEATURE_WIDTH:
            raise ValueError("leaf eval features have an unexpected width")
        if self.features.shape[0] != len(self.node_ids):
            raise ValueError("leaf eval feature rows must match node ids")


@dataclass(slots=True, frozen=True)
class LeafEvalBatchOutput:
    node_ids: tuple[int, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("leaf eval values must be a 2D tensor")
        if self.values.dtype != np.float32:
            raise ValueError("leaf eval values must use float32")
        if self.values.shape[1] != LEAF_EVAL_OUTPUT_WIDTH:
            raise ValueError("leaf eval values have an unexpected width")
        if self.values.shape[0] != len(self.node_ids):
            raise ValueError("leaf eval value rows must match node ids")


class LeafEvalBackend(Protocol):
    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        """Evaluate a fixed-size batch of leaf features."""

