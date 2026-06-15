from __future__ import annotations

import pytest

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import InfosetId, NodeType, PublicTree


def test_aggregate_prob_sum_preserves_node_reach_and_leaf_ids() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.LEAF,
            NodeType.TERMINAL,
        ),
        first_child=(0, 0, 0),
        child_count=(0, 0, 0),
        children=(),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, Chips(1)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.25),
        infoset_reach=(1.0,),
        action_reach=((1.0,), (), ()),
    )

    result = aggregate_prob_sum(tree, forward)

    assert result.node_reach_sum == (1.0, 0.5, 0.25)
    assert result.leaf_node_ids == (1,)
    assert result.leaf_reach_sum == (0.5,)
    assert result.leaf_batch.rows[0].node_id == 1
    assert result.leaf_batch.rows[0].reach == 0.5
    assert result.leaf_batch.rows[0].features == (0.5, 1.0)


def test_aggregate_prob_sum_rejects_mismatched_tree_and_forward_sizes() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(InfosetId(0),),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5),
        infoset_reach=(1.0,),
        action_reach=((1.0,),),
    )

    with pytest.raises(ValueError):
        aggregate_prob_sum(tree, forward)
