from __future__ import annotations

from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def make_leduc_public_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.CHANCE,
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 6, 8, 8, 8, 8, 8),
        child_count=(6, 2, 0, 0, 0, 0, 0),
        children=(
            ChildLink(child=NodeId(1), chance_prob=1 / 6),
            ChildLink(child=NodeId(2), chance_prob=1 / 6),
            ChildLink(child=NodeId(3), chance_prob=1 / 6),
            ChildLink(child=NodeId(4), chance_prob=1 / 6),
            ChildLink(child=NodeId(5), chance_prob=1 / 6),
            ChildLink(child=NodeId(6), chance_prob=1 / 6),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
        ),
        infoset_ids=(
            None,
            InfosetId(0),
            None,
            None,
            None,
            None,
            None,
        ),
        terminal_payoffs=(
            None,
            None,
            Chips(2),
            Chips(-1),
            Chips(4),
            Chips(-2),
            Chips(6),
        ),
    )
