from __future__ import annotations

from typing import Protocol

from .types import LeafFeatureBatch, LeafValueBatch


class LeafEvaluator(Protocol):
    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch: ...
