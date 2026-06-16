from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput


def test_leaf_eval_batch_input_validates_fixed_feature_layout() -> None:
    batch = LeafEvalBatchInput(
        node_ids=(2, 7),
        features=np.zeros((2, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32),
    )

    assert batch.node_ids == (2, 7)
    assert batch.features.shape == (2, LEAF_EVAL_FEATURE_WIDTH)


def test_leaf_eval_batch_output_validates_fixed_value_layout() -> None:
    batch = LeafEvalBatchOutput(
        node_ids=(2, 7),
        values=np.zeros((2, LEAF_EVAL_OUTPUT_WIDTH), dtype=np.float32),
    )

    assert batch.node_ids == (2, 7)
    assert batch.values.shape == (2, LEAF_EVAL_OUTPUT_WIDTH)


def test_leaf_eval_batch_rejects_mismatched_row_count() -> None:
    with pytest.raises(ValueError, match="feature rows must match node ids"):
        LeafEvalBatchInput(
            node_ids=(1,),
            features=np.zeros((2, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32),
        )

