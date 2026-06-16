from __future__ import annotations

from dataclasses import dataclass

from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH


@dataclass(slots=True, frozen=True)
class GpuLeafModelSpec:
    input_width: int = LEAF_EVAL_FEATURE_WIDTH
    output_width: int = LEAF_EVAL_OUTPUT_WIDTH
    hidden_widths: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    dtype: str = "float32"

