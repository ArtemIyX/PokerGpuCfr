import numpy as np

from pokergpu.cfr.traversal import build_leaf_feature_batch, scatter_leaf_values
from pokergpu.eval import CpuStubLeafEvaluator, LeafValueBatch
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_leaf_feature_batch_preserves_node_order() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF, NodeType.LEAF),
        is_frontier=(False, True, True),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, None),
    )

    batch = build_leaf_feature_batch(tree, (2, 1))

    assert batch.node_indices == (2, 1)
    assert batch.is_frontier.tolist() == [True, True]


def test_scatter_leaf_values_matches_input_order() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF, NodeType.LEAF),
        is_frontier=(False, True, True),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, None),
    )
    node_values_player0 = np.zeros(tree.node_count, dtype=np.float32)
    node_values_player1 = np.zeros(tree.node_count, dtype=np.float32)
    values = LeafValueBatch(
        ev_player0=np.array([1.5, -2.0], dtype=np.float32),
        ev_player1=np.array([-1.5, 2.0], dtype=np.float32),
    )

    scatter_leaf_values((2, 1), values, node_values_player0, node_values_player1)

    assert np.allclose(
        node_values_player0,
        np.array([0.0, -2.0, 1.5], dtype=np.float32),
    )
    assert np.allclose(
        node_values_player1,
        np.array([0.0, 2.0, -1.5], dtype=np.float32),
    )


def test_cpu_stub_leaf_evaluator_returns_batched_evs() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF, NodeType.LEAF),
        is_frontier=(False, True, True),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(ChildLink(NodeId(1)), ChildLink(NodeId(2))),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, None),
    )
    batch = build_leaf_feature_batch(tree, (1, 2))

    values = CpuStubLeafEvaluator().evaluate(batch)

    assert values.ev_player0.shape == (2,)
    assert values.ev_player1.shape == (2,)
