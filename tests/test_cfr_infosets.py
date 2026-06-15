from __future__ import annotations

import pytest

from pokergpu.cfr.solver import (
    build_dense_infoset_table,
    make_kuhn_public_tree,
    propagate_reach,
)
from pokergpu.tree.public_tree import (
    ChildLink,
    InfosetId,
    NodeId,
    NodeType,
    PublicTree,
)
from pokergpu.core.betting import Chips


def test_build_dense_infoset_table_extracts_dense_mappings() -> None:
    tree = make_kuhn_public_tree()

    table = build_dense_infoset_table(tree)

    assert table.infoset_count == 6
    assert table.infoset_to_node[0] == 1
    assert table.action_counts[0] == 2
    assert table.node_to_infoset[1] == 0
    assert table.infoset_nodes[0] == (1,)


def test_build_dense_infoset_table_rejects_missing_player_infoset() -> None:
    with pytest.raises(ValueError, match="player nodes must have infoset ids"):
        PublicTree(
            node_types=(NodeType.PLAYER0, NodeType.TERMINAL),
            first_child=(0, 0),
            child_count=(0, 0),
            children=(),
            infoset_ids=(None, None),
            terminal_payoffs=(None, Chips(1)),
        )


def test_build_dense_infoset_table_groups_repeated_infosets() -> None:
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

    table = build_dense_infoset_table(tree)
    reach = propagate_reach(
        tree,
        infoset_table=table,
        infoset_strategies={InfosetId(0): (0.25, 0.75), InfosetId(1): (1.0, 0.0)},
    )

    assert table.node_to_infoset == (0, 1, 0, -1, -1)
    assert table.infoset_nodes[0] == (0, 2)
    assert reach.infoset_reach[0] == 1.75
    assert reach.cumulative_strategy[0] == (0.4375, 1.3125)
