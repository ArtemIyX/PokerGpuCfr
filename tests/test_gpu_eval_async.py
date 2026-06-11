import numpy as np

from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.eval import AsyncLeafEvaluator
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_async_leaf_evaluator_returns_same_result() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF),
        is_frontier=(False, True),
        first_child=(0, 1),
        child_count=(1, 0),
        children=(ChildLink(NodeId(1)),),
        infoset_ids=(InfosetId(0), None),
        terminal_payoffs=(None, None),
    )
    batch = build_leaf_feature_batch(tree, (1,))
    async_eval = AsyncLeafEvaluator(CpuStubLeafEvaluator())

    result = async_eval.evaluate(batch)

    expected = CpuStubLeafEvaluator().evaluate(batch)
    assert np.allclose(result.ev_player0, expected.ev_player0)
    async_eval.close()
