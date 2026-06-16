from __future__ import annotations

import pytest
import numpy as np

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.stage2 import build_leaf_eval_batch
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

    assert result.node_aggregate.reach == (1.0, 0.5, 0.25)
    assert len(result.node_aggregate.card_reach) == 3
    assert len(result.node_aggregate.card_reach[0]) == 52
    assert len(result.node_aggregate.hand_reach) == 3
    assert len(result.node_aggregate.hand_reach[0]) == 1326
    assert result.node_aggregate.reach[0] == 1.0
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
    assert len(result.leaf_batch.rows[0].features.board_card_vector) == 52
    assert sum(result.leaf_batch.rows[0].features.board_card_vector) == 0.0
    assert len(result.leaf_batch.rows[0].features.leaf_card_reach_vector) == 52
    assert sum(result.leaf_batch.rows[0].features.leaf_card_reach_vector) == 0.0


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
    assert sum(result.leaf_batch.rows[0].features.board_card_vector) == 3.0
    assert sum(result.leaf_batch.rows[0].features.leaf_card_reach_vector) == pytest.approx(1.0)
    assert sum(
        1 for value in result.leaf_batch.rows[0].features.leaf_card_reach_vector if value > 0.0
    ) == 49
    assert sum(result.node_aggregate.card_reach[0]) == pytest.approx(1.0)
    assert sum(result.node_aggregate.hand_reach[0]) == pytest.approx(1.0)


def test_aggregate_prob_sum_blocks_board_cards_in_dense_vectors() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF, NodeType.LEAF),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(None, None),
        terminal_payoffs=(None, None),
    )
    forward = ForwardProfileResult(
        node_reach=(4.0, 8.0),
        infoset_reach=(),
        action_reach=((), ()),
    )

    result = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"))

    assert len(result.node_aggregate.card_reach) == 2
    assert len(result.node_aggregate.hand_reach) == 2
    assert sum(result.node_aggregate.card_reach[0]) == pytest.approx(4.0)
    assert sum(result.node_aggregate.card_reach[1]) == pytest.approx(8.0)
    assert result.node_aggregate.card_reach[0][8] == 0.0
    assert result.node_aggregate.card_reach[0][24] == 0.0
    assert result.node_aggregate.card_reach[0][38] == 0.0
    assert result.node_aggregate.card_reach[1][8] == 0.0
    assert result.node_aggregate.card_reach[1][24] == 0.0
    assert result.node_aggregate.card_reach[1][38] == 0.0
    assert sum(result.node_aggregate.hand_reach[0]) == pytest.approx(4.0)
    assert sum(result.node_aggregate.hand_reach[1]) == pytest.approx(8.0)


def test_node_card_aggregate_holds_node_and_card_reach() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(
        node_reach=(2.0,),
        infoset_reach=(),
        action_reach=((),),
    )

    result = aggregate_prob_sum(tree, forward)

    assert result.node_aggregate.reach == (2.0,)
    assert len(result.node_aggregate.card_reach) == 1
    assert len(result.node_aggregate.card_reach[0]) == 52
    assert sum(result.node_aggregate.card_reach[0]) == pytest.approx(2.0)
    assert len(result.node_aggregate.hand_reach) == 1
    assert len(result.node_aggregate.hand_reach[0]) == 1326
    assert sum(result.node_aggregate.hand_reach[0]) == pytest.approx(2.0)


def test_aggregate_prob_sum_parallel_node_card_reach() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.LEAF,
            NodeType.LEAF,
            NodeType.LEAF,
        ),
        first_child=(0, 0, 0),
        child_count=(0, 0, 0),
        children=(),
        infoset_ids=(None, None, None),
        terminal_payoffs=(None, None, None),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 2.0, 3.0),
        infoset_reach=(),
        action_reach=((), (), ()),
    )

    result = aggregate_prob_sum(tree, forward, max_workers=2)

    assert len(result.node_aggregate.card_reach) == 3
    assert sum(result.node_aggregate.card_reach[0]) == pytest.approx(1.0)
    assert sum(result.node_aggregate.card_reach[1]) == pytest.approx(2.0)
    assert sum(result.node_aggregate.card_reach[2]) == pytest.approx(3.0)
    assert sum(result.node_aggregate.hand_reach[0]) == pytest.approx(1.0)
    assert sum(result.node_aggregate.hand_reach[1]) == pytest.approx(2.0)
    assert sum(result.node_aggregate.hand_reach[2]) == pytest.approx(3.0)


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


def test_build_leaf_eval_batch_creates_fixed_width_tensor() -> None:
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

    aggregate = aggregate_prob_sum(tree, forward)
    batch = build_leaf_eval_batch(aggregate.leaf_batch)

    assert batch.node_ids == (0,)
    assert batch.features.shape == (1, LEAF_EVAL_FEATURE_WIDTH)
    assert batch.features.dtype == np.float32
    assert batch.features[0, 0] == pytest.approx(1.0)
    assert batch.features[0, 1] == pytest.approx(1.0)
