from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.leaf_eval import evaluate_leaf_batch


class _EchoBackend:
    def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        values = np.ones((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), dtype=np.float32)
        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)


def test_evaluate_leaf_batch_preserves_node_order() -> None:
    batch = LeafEvalBatchInput(
        node_ids=(3, 8),
        features=np.zeros((2, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32),
    )

    result = evaluate_leaf_batch(batch, _EchoBackend())

    assert result.node_ids == (3, 8)
    assert result.node_values == (1.0, 1.0)


def test_evaluate_leaf_batch_rejects_reordered_backend_output() -> None:
    class _BadBackend:
        def evaluate(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
            values = np.ones((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), dtype=np.float32)
            return LeafEvalBatchOutput(node_ids=tuple(reversed(batch.node_ids)), values=values)

    batch = LeafEvalBatchInput(
        node_ids=(3, 8),
        features=np.zeros((2, LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="preserve node ordering"):
        evaluate_leaf_batch(batch, _BadBackend())

