from __future__ import annotations

import pytest

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.core.betting import Chips
from pokergpu.core.board import Board
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
    assert result.leaf_batch.rows[0].features.reach == 0.5
    assert result.leaf_batch.rows[0].features.share == 1.0
    assert result.leaf_batch.rows[0].features.street == 0
    assert result.leaf_batch.rows[0].features.board_size == 0
    assert result.leaf_batch.rows[0].features.board_signature == 0
    assert len(result.leaf_batch.rows[0].features.board_card_mask) == 52
    assert not any(result.leaf_batch.rows[0].features.board_card_mask)


def test_aggregate_prob_sum_uses_board_street() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0,),
        infoset_reach=(),
        action_reach=((),),
    )

    result = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"))

    assert result.leaf_batch.rows[0].features.street == 1
    assert result.leaf_batch.rows[0].features.board_size == 3
    assert result.leaf_batch.rows[0].features.board_signature != 0
    assert sum(result.leaf_batch.rows[0].features.board_card_mask) == 3


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
