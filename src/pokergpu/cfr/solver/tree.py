from __future__ import annotations

from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def make_toy_public_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(1), Chips(-1)),
    )


def make_toy_pipeline_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.LEAF,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 3, 3),
        child_count=(2, 1, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), None, None),
        terminal_payoffs=(None, None, None, Chips(2)),
    )
