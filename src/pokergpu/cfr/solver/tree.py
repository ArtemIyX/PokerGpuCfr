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
    supported = ", ".join(
        (
            GameVariant.KUHN.value,
            GameVariant.LEDUC.value,
            GameVariant.HOLDEM_HU.value,
        )
    )
    raise NotImplementedError(f"unsupported game variant {game.value!r}; supported variants: {supported}")


def make_holdem_hu_public_tree() -> PublicTree:
    action_labels = ("check", "bet:25pct", "bet:50pct", "bet:75pct", "bet:100pct", "bet:150pct")
    child_count = len(action_labels)
    children = tuple(ChildLink(child=NodeId(index + 1)) for index in range(child_count))
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            *(NodeType.TERMINAL for _ in range(child_count)),
        ),
        first_child=(0, *([child_count] * child_count)),
        child_count=(child_count, *([0] * child_count)),
        children=children,
        infoset_ids=(InfosetId(0), *([None] * child_count)),
        terminal_payoffs=(None, *(Chips(0) for _ in range(child_count))),
        action_labels=(action_labels, *([None] * child_count)),
    )


def resolve_game_state_spec(spec: GameStateSpec | None) -> GameStateSpec | None:
    if spec is None:
        return None
    if spec.mode is GameStateMode.EXACT:
        return spec
    if spec.mode is GameStateMode.RANDOM:
        return spec
    raise ValueError(f"unsupported game state mode: {spec.mode}")
