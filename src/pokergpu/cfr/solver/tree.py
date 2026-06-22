from __future__ import annotations

from pokergpu.core.betting import Chips
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
    raise NotImplementedError(f"public tree factory not implemented for {game}")


def make_holdem_hu_public_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.TERMINAL,
            NodeType.LEAF,
        ),
        first_child=(0, 2, 4, 4),
        child_count=(2, 2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(2)),
            ChildLink(child=NodeId(3)),
        ),
        infoset_ids=(
            InfosetId(0),
            InfosetId(1),
            None,
            None,
        ),
        terminal_payoffs=(
            None,
            None,
            Chips(1),
            None,
        ),
    )


def resolve_game_state_spec(spec: GameStateSpec | None) -> GameStateSpec | None:
    if spec is None:
        return None
    if spec.mode is GameStateMode.EXACT:
        return spec
    if spec.mode is GameStateMode.RANDOM:
        return spec
    raise ValueError(f"unsupported game state mode: {spec.mode}")
