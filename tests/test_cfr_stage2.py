from __future__ import annotations

import importlib

import pytest
import numpy as np

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH
from pokergpu.cfr.stage2 import build_leaf_eval_batch
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage2 import aggregate_prob_sum_prepacked
from pokergpu.cfr.stage2 import prepare_stage2_input
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
    assert result.node_aggregate.card_reach.shape == (3, 52)
    assert result.node_aggregate.card_reach.dtype == np.float64
    assert result.node_aggregate.hand_reach.shape == (3, 1326)
    assert result.node_aggregate.hand_reach.dtype == np.float64
    assert result.node_aggregate.reach[0] == 1.0
    assert result.leaf_node_ids == (1,)
    assert result.leaf_reach_sum == (0.5,)
    assert result.leaf_batch.node_ids == (1,)
    assert result.leaf_batch.reach.shape == (1,)
    assert result.leaf_batch.reach.dtype == np.float32
    assert result.leaf_batch.reach[0] == pytest.approx(0.5)
    assert result.leaf_batch.features.shape == (1, LEAF_EVAL_FEATURE_WIDTH)
    assert result.leaf_batch.features.dtype == np.float32
    assert result.leaf_batch.features[0, 0] == pytest.approx(0.5)
    assert result.leaf_batch.features[0, 1] == pytest.approx(1.0)
    assert result.leaf_batch.features[0, 2] == pytest.approx(0.0)


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

    assert result.leaf_batch.features.shape == (1, LEAF_EVAL_FEATURE_WIDTH)
    assert result.leaf_batch.features[0, 0] == pytest.approx(1.0)
    assert result.leaf_batch.features[0, 1] == pytest.approx(1.0)
    assert sum(
        1 for value in result.leaf_batch.features[0, 109:161] if value > 0.0
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

    assert result.node_aggregate.card_reach.shape == (2, 52)
    assert result.node_aggregate.hand_reach.shape == (2, 1326)
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


def test_aggregate_prob_sum_handles_tree_without_leaf_nodes() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(InfosetId(0),),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0,),
        infoset_reach=(1.0,),
        action_reach=((),),
    )

    result = aggregate_prob_sum(tree, forward)

    assert result.leaf_node_ids == ()
    assert result.leaf_reach_sum.shape == (0,)
    assert result.leaf_batch.node_ids == ()
    assert result.leaf_batch.reach.shape == (0,)
    assert result.leaf_batch.features.shape == (0, LEAF_EVAL_FEATURE_WIDTH)
    assert result.leaf_batch.features.dtype == np.float32


def test_aggregate_prob_sum_builds_direct_leaf_tensor_for_boarded_state() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF, NodeType.LEAF),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(None, None),
        terminal_payoffs=(None, None),
    )
    forward = ForwardProfileResult(
        node_reach=(3.0, 1.5),
        infoset_reach=(),
        action_reach=((), ()),
    )

    result = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"))

    assert result.leaf_batch.node_ids == (0, 1)
    assert result.leaf_batch.reach.shape == (2,)
    assert result.leaf_batch.features.shape == (2, LEAF_EVAL_FEATURE_WIDTH)
    assert result.leaf_batch.features.dtype == np.float32
    assert result.leaf_batch.features[0, 0] == pytest.approx(3.0)
    assert result.leaf_batch.features[1, 0] == pytest.approx(1.5)
    assert result.leaf_batch.features[0, 1] == pytest.approx(2.0 / 3.0)
    assert result.leaf_batch.features[1, 1] == pytest.approx(1.0 / 3.0)
    assert result.leaf_batch.features[0, 2] == pytest.approx(1.0)
    assert result.leaf_batch.features[0, 3] == pytest.approx(3.0)
    assert result.leaf_batch.features[0, 4] != 0.0


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
    assert result.node_aggregate.card_reach.shape == (1, 52)
    assert result.node_aggregate.card_reach.dtype == np.float64
    assert sum(result.node_aggregate.card_reach[0]) == pytest.approx(2.0)
    assert result.node_aggregate.hand_reach.shape == (1, 1326)
    assert result.node_aggregate.hand_reach.dtype == np.float64
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

    assert result.node_aggregate.card_reach.shape == (3, 52)
    assert sum(result.node_aggregate.card_reach[0]) == pytest.approx(1.0)
    assert sum(result.node_aggregate.card_reach[1]) == pytest.approx(2.0)
    assert sum(result.node_aggregate.card_reach[2]) == pytest.approx(3.0)
    assert sum(result.node_aggregate.hand_reach[0]) == pytest.approx(1.0)
    assert sum(result.node_aggregate.hand_reach[1]) == pytest.approx(2.0)
    assert sum(result.node_aggregate.hand_reach[2]) == pytest.approx(3.0)


def test_aggregate_prob_sum_parallel_matches_serial_dense_outputs() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.LEAF,
            NodeType.LEAF,
            NodeType.LEAF,
            NodeType.LEAF,
        ),
        first_child=(0, 0, 0, 0),
        child_count=(0, 0, 0, 0),
        children=(),
        infoset_ids=(None, None, None, None),
        terminal_payoffs=(None, None, None, None),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 2.0, 3.0, 4.0),
        infoset_reach=(),
        action_reach=((), (), (), ()),
    )

    serial = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"))
    parallel = aggregate_prob_sum(tree, forward, Board.from_str("AhKdTc"), max_workers=2)

    assert np.allclose(serial.node_aggregate.card_reach, parallel.node_aggregate.card_reach)
    assert np.allclose(serial.node_aggregate.hand_reach, parallel.node_aggregate.hand_reach)
    assert np.allclose(serial.leaf_batch.reach, parallel.leaf_batch.reach)
    assert np.allclose(serial.leaf_batch.features, parallel.leaf_batch.features)


def test_aggregate_prob_sum_numba_path_matches_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    stage2 = importlib.import_module("pokergpu.cfr.stage2")
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
    board = Board.from_str("AhKdTc")

    serial = stage2.aggregate_prob_sum(tree, forward, board, max_workers=2)

    monkeypatch.setenv("POKERGPU_STAGE2_NUMBA", "1")
    reloaded = importlib.reload(stage2)
    numba_result = reloaded.aggregate_prob_sum(tree, forward, board, max_workers=2)

    assert np.allclose(serial.node_aggregate.card_reach, numba_result.node_aggregate.card_reach)
    assert np.allclose(serial.node_aggregate.hand_reach, numba_result.node_aggregate.hand_reach)
    assert np.allclose(serial.leaf_batch.reach, numba_result.leaf_batch.reach)
    assert np.allclose(serial.leaf_batch.features, numba_result.leaf_batch.features)


def test_aggregate_prob_sum_prepacked_matches_wrapper() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF, NodeType.LEAF),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(None, None),
        terminal_payoffs=(None, None),
    )
    forward = ForwardProfileResult(
        node_reach=(2.5, 1.25),
        infoset_reach=(),
        action_reach=((), ()),
    )
    board = Board.from_str("AhKdTc")

    prepared = prepare_stage2_input(tree, board, forward)
    wrapper = aggregate_prob_sum(tree, forward, board, max_workers=2)
    prepacked = aggregate_prob_sum_prepacked(prepared, max_workers=2)

    assert np.allclose(wrapper.node_aggregate.card_reach, prepacked.node_aggregate.card_reach)
    assert np.allclose(wrapper.node_aggregate.hand_reach, prepacked.node_aggregate.hand_reach)
    assert np.allclose(wrapper.leaf_batch.reach, prepacked.leaf_batch.reach)
    assert np.allclose(wrapper.leaf_batch.features, prepacked.leaf_batch.features)


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
