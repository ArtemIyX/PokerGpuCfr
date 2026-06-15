from __future__ import annotations

import pytest

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import build_showdown_equity_board_cache
from pokergpu.cfr.stage4 import build_showdown_equity_input, compute_showdown_equity
from pokergpu.core.betting import Chips
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_build_showdown_equity_input_is_node_local() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.TERMINAL,
        ),
        first_child=(0, 1, 0),
        child_count=(1, 1, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), None),
        terminal_payoffs=(None, None, Chips(0)),
    )
    board = Board.from_str("AhKdTc9s2c")
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.0),
        infoset_reach=(1.0, 0.5),
        action_reach=((1.0,), (1.0,), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward, board)
    opponent = compute_opponent_reach(tree, aggregate)

    result = build_showdown_equity_input(tree, aggregate, opponent, board=board)

    assert len(result.rows) == 3
    assert all(row.board == board for row in result.rows)
    assert result.rows[0].node_id == 0
    assert result.rows[1].node_id == 1
    assert len(result.rows[0].opponent_reach) == 1326
    assert len(result.rows[0].live_hand_mask) == 1326
    assert result.rows[0].pot_size == pytest.approx(1.0)


def test_build_showdown_equity_board_cache_precomputes_board_only_data() -> None:
    board = Board.from_str("AhKdTc9s2c")

    cache = build_showdown_equity_board_cache(board)

    assert cache.board == board
    assert len(cache.live_hand_mask) == 1326
    assert len(cache.hand_scores) == 1326
    assert len(cache.live_hand_indices) < 1326
    assert len(cache.comparison_matrix) == 1326
    assert len(cache.comparison_matrix[0]) == 1326
    assert sum(cache.live_hand_mask) == len(cache.live_hand_indices)
    assert cache.comparison_matrix[0][0] == 0.0


def test_compute_showdown_equity_returns_node_aligned_output() -> None:
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    board = Board.from_str("AhKdTc9s2c")
    forward = ForwardProfileResult(
        node_reach=(1.0,),
        infoset_reach=(),
        action_reach=((),),
    )
    aggregate = aggregate_prob_sum(tree, forward, board)
    opponent = compute_opponent_reach(tree, aggregate)

    result = compute_showdown_equity(tree, aggregate, opponent, board=board)

    assert len(result.node_showdown_equity) == 1
    assert len(result.node_showdown_equity_bb) == 1
    assert result.output_rows[0].node_id == 0
    assert result.output_rows[0].showdown_equity == result.node_showdown_equity[0]
    assert result.output_rows[0].showdown_equity_bb == result.node_showdown_equity_bb[0]


def test_compute_showdown_equity_has_exact_single_hand_showdown_value() -> None:
    board = Board.from_str("AhKdTc9s2c")
    tree = PublicTree(
        node_types=(NodeType.LEAF,),
        first_child=(0,),
        child_count=(0,),
        children=(),
        infoset_ids=(None,),
        terminal_payoffs=(None,),
    )
    forward = ForwardProfileResult(node_reach=(1.0,), infoset_reach=(), action_reach=((),))
    aggregate = aggregate_prob_sum(tree, forward, board)
    opponent = compute_opponent_reach(tree, aggregate)

    result = compute_showdown_equity(
        tree,
        aggregate,
        opponent,
        board=board,
    )

    assert result.node_showdown_equity[0] == pytest.approx(result.node_showdown_equity_bb[0])
    assert 0.0 <= result.node_showdown_equity[0] <= 1.0


def test_compute_showdown_equity_rejects_preflop_board() -> None:
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
    opponent = compute_opponent_reach(tree, aggregate)

    with pytest.raises(ValueError, match="river board"):
        compute_showdown_equity(tree, aggregate, opponent)
