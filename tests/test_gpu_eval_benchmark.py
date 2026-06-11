from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.eval import measure_leaf_batch_throughput
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_measure_leaf_batch_throughput_returns_positive_rate() -> None:
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

    result = measure_leaf_batch_throughput(
        CpuStubLeafEvaluator(),
        batch,
        repeats=2,
    )

    assert result.batch_size == 1
    assert result.elapsed_seconds >= 0.0
    assert result.leaves_per_second > 0.0
