from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.solver.infosets import build_dense_infoset_table
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_compute_opponent_reach_aggregates_infoset_reach() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 1, 0, 0),
        child_count=(1, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(0), None, None),
        terminal_payoffs=(None, None, Chips(0), Chips(0)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.4, 0.0, 0.0),
        infoset_reach=(1.4,),
        action_reach=((0.4,), (0.25, 0.75), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)

    result = compute_opponent_reach(tree, aggregate)

    assert result.infoset_opponent_reach == (1.4,)
    assert len(result.infoset_card_opponent_reach) == 1
    assert len(result.infoset_card_opponent_reach[0]) == 52
    assert sum(result.infoset_card_opponent_reach[0]) == pytest.approx(1.4)
    assert len(result.infoset_hand_opponent_reach) == 1
    assert len(result.infoset_hand_opponent_reach[0]) == 1326
    assert len(result.infoset_node_hand_ratio) == 1
    assert len(result.infoset_node_hand_ratio[0]) == 2
    assert len(result.infoset_node_hand_ratio[0][0]) == 1326
    assert (
        result.infoset_node_hand_ratio[0][0][0] + result.infoset_node_hand_ratio[0][1][0]
    ) == pytest.approx(1.0)
    assert len(result.infoset_node_card_ratio) == 1
    assert len(result.infoset_node_card_ratio[0]) == 2
    assert len(result.infoset_node_card_ratio[0][0]) == 52
    assert len(result.node_hand_opponent_reach) == 4
    assert len(result.node_hand_opponent_reach[0]) == 1326
    assert all(
        total == pytest.approx(1.0)
        for total in (
            result.infoset_node_card_ratio[0][0][0] + result.infoset_node_card_ratio[0][1][0],
            result.infoset_node_card_ratio[0][0][1] + result.infoset_node_card_ratio[0][1][1],
        )
    )
    assert result.node_opponent_reach == (1.0, 0.4, 0.0, 0.0)
    assert result.node_opponent_share == pytest.approx((5 / 7, 2 / 7, 0.0, 0.0))


def test_compute_opponent_reach_uses_uniform_shares_for_zero_reach_infoset() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.PLAYER0),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(InfosetId(0), InfosetId(0)),
        terminal_payoffs=(None, None),
    )
    aggregate = aggregate_prob_sum(
        tree,
        ForwardProfileResult(
            node_reach=(0.0, 0.0),
            infoset_reach=(0.0,),
            action_reach=((), ()),
        ),
    )

    result = compute_opponent_reach(tree, aggregate, max_workers=2)

    assert result.infoset_opponent_reach == (0.0,)
    assert len(result.infoset_card_opponent_reach) == 1
    assert sum(result.infoset_card_opponent_reach[0]) == 0.0
    assert len(result.infoset_hand_opponent_reach[0]) == 1326
    assert len(result.infoset_node_hand_ratio[0]) == 2
    assert len(result.infoset_node_card_ratio[0]) == 2
    assert result.infoset_node_hand_ratio[0][0][0] == 0.0
    assert sum(result.infoset_node_card_ratio[0][0]) == 0.0
    assert len(result.node_hand_opponent_reach) == 2
    assert result.node_opponent_share == (0.5, 0.5)


def test_compute_opponent_reach_rejects_non_dense_infosets() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.PLAYER0),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(InfosetId(0), InfosetId(2)),
        terminal_payoffs=(None, None),
    )
    aggregate = aggregate_prob_sum(
        tree,
        ForwardProfileResult(
            node_reach=(1.0, 1.0),
            infoset_reach=(1.0, 0.0, 1.0),
            action_reach=((), ()),
        ),
    )

    with pytest.raises(ValueError, match="infoset ids must be dense and contiguous"):
        compute_opponent_reach(tree, aggregate)


def test_compute_opponent_reach_handles_repeated_infosets() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 4, 4, 4),
        child_count=(2, 2, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), InfosetId(0), None, None),
        terminal_payoffs=(None, None, None, Chips(1), Chips(2)),
    )
    aggregate = aggregate_prob_sum(
        tree,
        ForwardProfileResult(
            node_reach=(1.0, 0.5, 0.25, 0.0, 0.0),
            infoset_reach=(1.25, 0.5),
            action_reach=((0.4, 0.6), (0.5, 0.5), (0.2, 0.8), (), ()),
        ),
    )

    result = compute_opponent_reach(tree, aggregate)

    assert result.infoset_opponent_reach == (1.25, 0.5)
    assert len(result.infoset_card_opponent_reach) == 2
    assert sum(result.infoset_card_opponent_reach[0]) == pytest.approx(1.25)
    assert sum(result.infoset_card_opponent_reach[1]) == pytest.approx(0.5)
    assert len(result.infoset_hand_opponent_reach[0]) == 1326
    assert len(result.infoset_node_hand_ratio[0]) == 2
    assert len(result.infoset_node_hand_ratio[1]) == 1
    assert result.infoset_node_hand_ratio[0][0][0] + result.infoset_node_hand_ratio[0][1][0] == pytest.approx(1.0)
    assert len(result.infoset_node_card_ratio[0]) == 2
    assert len(result.infoset_node_card_ratio[1]) == 1
    assert result.infoset_node_card_ratio[0][0][0] + result.infoset_node_card_ratio[0][1][0] == pytest.approx(1.0)
    assert len(result.node_hand_opponent_reach) == 5
    assert result.node_opponent_reach == (1.0, 0.5, 0.25, 0.0, 0.0)
    assert result.node_opponent_share == pytest.approx((0.8, 1.0, 0.2, 0.0, 0.0))


def test_compute_opponent_reach_matches_reference_baseline_on_tiny_tree() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.PLAYER0),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(InfosetId(0), InfosetId(0)),
        terminal_payoffs=(None, None),
    )
    forward = ForwardProfileResult(
        node_reach=(2.0, 4.0),
        infoset_reach=(6.0,),
        action_reach=((), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    node0_hand = [0.0 for _ in range(1326)]
    node1_hand = [0.0 for _ in range(1326)]
    node0_hand[0] = 2.0
    node0_hand[1] = 1.0
    node1_hand[0] = 1.0
    node1_hand[1] = 3.0
    object.__setattr__(
        aggregate.node_aggregate,
        "hand_reach",
        (
            tuple(node0_hand),
            tuple(node1_hand),
        ),
    )

    result = compute_opponent_reach(tree, aggregate)

    assert result.infoset_opponent_reach == (6.0,)
    assert result.infoset_hand_opponent_reach[0][0] == pytest.approx(3.0)
    assert result.infoset_hand_opponent_reach[0][1] == pytest.approx(4.0)
    assert result.infoset_node_hand_ratio[0][0][0] == pytest.approx(2.0 / 3.0)
    assert result.infoset_node_hand_ratio[0][1][0] == pytest.approx(1.0 / 3.0)
    assert result.infoset_node_hand_ratio[0][0][1] == pytest.approx(1.0 / 4.0)
    assert result.infoset_node_hand_ratio[0][1][1] == pytest.approx(3.0 / 4.0)
    assert result.node_hand_opponent_reach[0][0] == pytest.approx(2.0 / 3.0)
    assert result.node_hand_opponent_reach[1][0] == pytest.approx(1.0 / 3.0)
    assert result.node_hand_opponent_reach[0][1] == pytest.approx(1.0 / 4.0)
    assert result.node_hand_opponent_reach[1][1] == pytest.approx(3.0 / 4.0)


def test_compute_opponent_reach_matches_reference_baseline_with_repeated_infosets() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 4, 4, 4),
        child_count=(2, 2, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(4)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), InfosetId(0), None, None),
        terminal_payoffs=(None, None, None, Chips(1), Chips(2)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.25, 0.0, 0.0),
        infoset_reach=(1.25, 0.5),
        action_reach=((0.4, 0.6), (0.5, 0.5), (0.2, 0.8), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    node0_hand = [0.0 for _ in range(1326)]
    node1_hand = [0.0 for _ in range(1326)]
    node2_hand = [0.0 for _ in range(1326)]
    node0_hand[0] = 1.0
    node0_hand[1] = 2.0
    node1_hand[0] = 3.0
    node1_hand[1] = 1.0
    node2_hand[0] = 2.0
    node2_hand[1] = 4.0
    object.__setattr__(
        aggregate.node_aggregate,
        "hand_reach",
        (
            tuple(node0_hand),
            tuple(node1_hand),
            tuple(node2_hand),
            tuple(0.0 for _ in range(1326)),
            tuple(0.0 for _ in range(1326)),
        ),
    )

    result = compute_opponent_reach(tree, aggregate)

    assert result.infoset_opponent_reach == (1.25, 0.5)
    assert result.infoset_hand_opponent_reach[0][0] == pytest.approx(3.0)
    assert result.infoset_hand_opponent_reach[0][1] == pytest.approx(6.0)
    assert result.infoset_node_hand_ratio[0][0][0] == pytest.approx(1.0 / 3.0)
    assert result.infoset_node_hand_ratio[0][1][0] == pytest.approx(2.0 / 3.0)
    assert result.infoset_node_hand_ratio[0][0][1] == pytest.approx(2.0 / 6.0)
    assert result.infoset_node_hand_ratio[0][1][1] == pytest.approx(4.0 / 6.0)


def test_compute_opponent_reach_accepts_numpy_hand_reach_from_stage2() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.PLAYER0),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(InfosetId(0), InfosetId(0)),
        terminal_payoffs=(None, None),
    )
    aggregate = aggregate_prob_sum(
        tree,
        ForwardProfileResult(
            node_reach=(1.0, 2.0),
            infoset_reach=(3.0,),
            action_reach=((), ()),
        ),
    )

    result = compute_opponent_reach(tree, aggregate)

    assert isinstance(aggregate.node_aggregate.hand_reach, np.ndarray)
    assert aggregate.node_aggregate.hand_reach.shape == (2, 1326)
    assert result.infoset_opponent_reach == (3.0,)
    assert len(result.infoset_hand_opponent_reach[0]) == 1326
    assert len(result.node_hand_opponent_reach[0]) == 1326


def test_compute_opponent_reach_accepts_tuple_hand_reach_compatibility() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.PLAYER0),
        first_child=(0, 0),
        child_count=(0, 0),
        children=(),
        infoset_ids=(InfosetId(0), InfosetId(0)),
        terminal_payoffs=(None, None),
    )
    aggregate = aggregate_prob_sum(
        tree,
        ForwardProfileResult(
            node_reach=(2.0, 4.0),
            infoset_reach=(6.0,),
            action_reach=((), ()),
        ),
    )
    node0_hand = [0.0 for _ in range(1326)]
    node1_hand = [0.0 for _ in range(1326)]
    node0_hand[0] = 2.0
    node1_hand[0] = 1.0
    object.__setattr__(
        aggregate.node_aggregate,
        "hand_reach",
        (
            tuple(node0_hand),
            tuple(node1_hand),
        ),
    )

    result = compute_opponent_reach(tree, aggregate)

    assert result.infoset_opponent_reach == (6.0,)
    assert result.infoset_hand_opponent_reach[0][0] == pytest.approx(3.0)
    assert result.node_hand_opponent_reach[0][0] == pytest.approx(2.0 / 3.0)


def test_build_dense_infoset_table_is_cached_for_same_tree() -> None:
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.PLAYER1, NodeType.TERMINAL),
        first_child=(0, 1, 0),
        child_count=(1, 1, 0),
        children=(ChildLink(child=NodeId(1)), ChildLink(child=NodeId(2))),
        infoset_ids=(InfosetId(0), InfosetId(1), None),
        terminal_payoffs=(None, None, Chips(0)),
    )

    first = build_dense_infoset_table(tree)
    second = build_dense_infoset_table(tree)

    assert first is second
    assert first.infoset_nodes == ((0,), (1,))
