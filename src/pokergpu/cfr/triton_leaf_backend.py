from __future__ import annotations

from dataclasses import dataclass

from pokergpu.cfr.leaf_model_spec import GpuLeafModelSpec
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput


@dataclass(slots=True, frozen=True)
class TritonLeafKernel:
    spec: GpuLeafModelSpec = GpuLeafModelSpec()

    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        raise NotImplementedError("Triton leaf kernel is not implemented yet")
