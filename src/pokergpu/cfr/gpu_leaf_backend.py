from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.leaf_eval import LeafEvalBackend


@dataclass(slots=True, frozen=True)
class GpuLeafModelSpec:
    input_width: int = LEAF_EVAL_FEATURE_WIDTH
    output_width: int = LEAF_EVAL_OUTPUT_WIDTH
    hidden_widths: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    dtype: str = "float32"


class GpuLeafKernel(Protocol):
    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        """Evaluate a fixed leaf batch on GPU and return batched outputs."""


@dataclass(slots=True, frozen=True)
class GpuLeafBackend(LeafEvalBackend):
    kernel: GpuLeafKernel
    spec: GpuLeafModelSpec = GpuLeafModelSpec()

    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        if batch.features.shape[1] != self.spec.input_width:
            raise ValueError("gpu leaf backend received an unexpected input width")
        values = self.kernel(batch)
        if values.node_ids != batch.node_ids:
            raise ValueError("gpu leaf backend must preserve node ordering")
        return values
