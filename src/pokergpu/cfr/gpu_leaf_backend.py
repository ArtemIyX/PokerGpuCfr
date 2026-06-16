from __future__ import annotations

from dataclasses import dataclass

from pokergpu.cfr.leaf_eval import LeafEvalBackend
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput


@dataclass(slots=True, frozen=True)
class UnavailableGpuLeafBackend:
    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        raise NotImplementedError(
            "GPU leaf evaluation backend is not implemented yet"
        )


def create_default_leaf_backend() -> LeafEvalBackend:
    return UnavailableGpuLeafBackend()

