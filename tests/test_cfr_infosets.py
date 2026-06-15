from __future__ import annotations

import pytest

from pokergpu.cfr.solver import build_dense_infoset_table, make_kuhn_public_tree
from pokergpu.tree.public_tree import NodeType, PublicTree
from pokergpu.core.betting import Chips


def test_build_dense_infoset_table_extracts_dense_mappings() -> None:
    tree = make_kuhn_public_tree()

    table = build_dense_infoset_table(tree)

    assert table.infoset_count == 6
    assert table.infoset_to_node[0] == 1
    assert table.action_counts[0] == 2
    assert table.node_to_infoset[1] == 0


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
