from __future__ import annotations

from pokergpu.cfr.stage1 import normalize_strategy, propagate_forward
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_normalize_strategy_falls_back_to_uniform() -> None:
    assert normalize_strategy((0.0, 0.0, 0.0)) == (1 / 3, 1 / 3, 1 / 3)


def test_normalize_strategy_clips_negative_weights() -> None:
    assert normalize_strategy((-1.0, 3.0)) == (0.0, 1.0)


def test_propagate_forward_distributes_reach_through_tree() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2, 2),
        child_count=(2, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), None, None),
        terminal_payoffs=(None, None, Chips(10), Chips(-10)),
    )

    result = propagate_forward(
        tree,
        infoset_strategies={
            InfosetId(0): (0.25, 0.75),
            InfosetId(1): (0.6, 0.4),
        },
    )

    assert result.node_reach == (1.0, 0.25, 0.9, 0.1)
    assert result.infoset_reach == (1.0, 0.25)
    assert result.action_reach[0] == (0.25, 0.75)


def test_propagate_forward_propagates_chance_nodes() -> None:
    tree = PublicTree(
        node_types=(NodeType.CHANCE, NodeType.TERMINAL, NodeType.TERMINAL),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1), chance_prob=0.25),
            ChildLink(child=NodeId(2), chance_prob=0.75),
        ),
        infoset_ids=(None, None, None),
        terminal_payoffs=(None, Chips(0), Chips(0)),
    )

    result = propagate_forward(tree)

    assert result.node_reach == (1.0, 0.25, 0.75)
