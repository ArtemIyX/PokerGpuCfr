from __future__ import annotations

from pokergpu.core.betting import Chips
from pokergpu.abstraction.actions import action_labels_for_street
from pokergpu.abstraction.actions import make_holdem_hu_profile
from pokergpu.core.board import Street
from pokergpu.cfr.solver.spec import GameStateMode
from pokergpu.cfr.solver.spec import GameStateSpec
from pokergpu.cfr.solver.spec import GameVariant
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
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


def make_game_public_tree(game: GameVariant) -> PublicTree:
    if game is GameVariant.KUHN:
        return make_kuhn_public_tree()
    if game is GameVariant.LEDUC:
        return make_leduc_public_tree()
    if game is GameVariant.HOLDEM_HU:
        return make_holdem_hu_public_tree()
    supported = ", ".join(
        (
            GameVariant.KUHN.value,
            GameVariant.LEDUC.value,
            GameVariant.HOLDEM_HU.value,
        )
    )
    raise NotImplementedError(f"unsupported game variant {game.value!r}; supported variants: {supported}")


def make_holdem_hu_public_tree() -> PublicTree:
    profile = make_holdem_hu_profile()
    action_labels = action_labels_for_street(profile, Street.PREFLOP)
    action_count = len(action_labels)
    node_types = (
        NodeType.PLAYER0,
        NodeType.TERMINAL,
        NodeType.TERMINAL,
        NodeType.TERMINAL,
        NodeType.TERMINAL,
        NodeType.TERMINAL,
        NodeType.LEAF,
    )
    children = (
        ChildLink(child=NodeId(1)),
        ChildLink(child=NodeId(2)),
        ChildLink(child=NodeId(3)),
        ChildLink(child=NodeId(4)),
        ChildLink(child=NodeId(5)),
        ChildLink(child=NodeId(6)),
    )
    first_child = (0, *(action_count for _ in range(action_count)))
    child_counts = (action_count, *(0 for _ in range(action_count)))
    infosets = (InfosetId(0), *(None for _ in range(action_count)))
    terminals = (
        None,
        Chips(-3),
        Chips(-2),
        Chips(-1),
        Chips(0),
        Chips(1),
        None,
    )
    return PublicTree(
        node_types=node_types,
        first_child=first_child,
        child_count=child_counts,
        children=children,
        infoset_ids=infosets,
        terminal_payoffs=terminals,
        action_labels=(action_labels, *(None for _ in range(action_count))),
    )


def resolve_game_state_spec(spec: GameStateSpec | None) -> GameStateSpec | None:
    if spec is None:
        return None
    if spec.mode is GameStateMode.EXACT:
        return spec
    if spec.mode is GameStateMode.RANDOM:
        return spec
    raise ValueError(f"unsupported game state mode: {spec.mode}")
