from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from .interface import LeafEvaluator
from .types import LeafFeatureBatch, LeafValueBatch


class AsyncLeafEvaluator:
    def __init__(
        self,
        evaluator: LeafEvaluator,
        max_workers: int = 1,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._evaluator = evaluator
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, batch: LeafFeatureBatch) -> Future[LeafValueBatch]:
        return self._executor.submit(self._evaluator.evaluate, batch)

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        return self.submit(batch).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
