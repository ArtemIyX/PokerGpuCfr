from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


LEAF_EVAL_FEATURE_WIDTH = 162
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
        if not np.isfinite(self.values).all():
            raise ValueError("leaf eval values must be finite")


class LeafEvalBackend(Protocol):
    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        """Evaluate a fixed-size batch of leaf features."""


@dataclass(slots=True, frozen=True)
class LeafEvalResult:
    node_ids: tuple[int, ...]
    node_values: tuple[float, ...]


def evaluate_leaf_batch(
    batch: LeafEvalBatchInput,
    backend: LeafEvalBackend,
) -> LeafEvalResult:
    output = backend.evaluate(batch)
    if output.node_ids != batch.node_ids:
        raise ValueError("leaf eval backend must preserve node ordering")
    if output.values.shape[0] != len(batch.node_ids):
        raise ValueError("leaf eval backend returned an unexpected row count")
    if output.values.dtype != np.float32:
        raise ValueError("leaf eval backend must return float32 values")
    if not np.isfinite(output.values).all():
        raise ValueError("leaf eval backend returned non-finite values")
    values = tuple(float(value) for value in output.values[:, 0])
    return LeafEvalResult(node_ids=output.node_ids, node_values=values)
