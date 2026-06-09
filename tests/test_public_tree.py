import pytest

from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_public_tree_exposes_child_ranges() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.LEAF,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(100), None),
    )

    assert tree.node_count == 3
    assert tree.child_links(NodeId(0)) == (
        ChildLink(child=NodeId(1)),
        ChildLink(child=NodeId(2)),
    )


def test_public_tree_validates_chance_probabilities() -> None:
    with pytest.raises(ValueError):
        PublicTree(
            node_types=(NodeType.CHANCE, NodeType.TERMINAL, NodeType.TERMINAL),
            first_child=(0, 2, 2),
            child_count=(2, 0, 0),
            children=(
                ChildLink(child=NodeId(1), chance_prob=0.3),
                ChildLink(child=NodeId(2), chance_prob=0.3),
            ),
            infoset_ids=(None, None, None),
            terminal_payoffs=(None, Chips(10), Chips(20)),
        )


def test_public_tree_rejects_missing_infoset_on_player_node() -> None:
    with pytest.raises(ValueError):
        PublicTree(
            node_types=(NodeType.PLAYER0,),
            first_child=(0,),
            child_count=(0,),
            children=(),
            infoset_ids=(None,),
            terminal_payoffs=(None,),
        )


def test_public_tree_rejects_payoff_on_non_terminal_node() -> None:
    with pytest.raises(ValueError):
        PublicTree(
            node_types=(NodeType.LEAF,),
            first_child=(0,),
            child_count=(0,),
            children=(),
            infoset_ids=(None,),
            terminal_payoffs=(Chips(10),),
        )
