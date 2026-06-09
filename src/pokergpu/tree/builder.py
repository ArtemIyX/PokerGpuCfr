from __future__ import annotations

from dataclasses import dataclass

from pokergpu.abstraction.actions import ActionAbstraction, BaselineActionAbstraction
from pokergpu.core.actions import Action
from pokergpu.core.betting import Chips
from pokergpu.core.state import GameState
from pokergpu.core.transitions import apply_action

from .public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class BuiltPublicTree:
    tree: PublicTree
    node_states: tuple[GameState, ...]
    actions_by_node: tuple[tuple[Action, ...], ...]


def build_shallow_public_tree(
    state: GameState,
    *,
    abstraction: ActionAbstraction | None = None,
) -> BuiltPublicTree:
    abstraction_impl = abstraction or BaselineActionAbstraction()
    root_actions = abstraction_impl.legal_actions(state)

    root_node_type = (
        NodeType.PLAYER0 if state.betting_round.to_act == 0 else NodeType.PLAYER1
    )
    node_types = [root_node_type]
    first_child = [0]
    child_count = [len(root_actions)]
    infoset_ids: list[InfosetId | None] = [InfosetId(0)]
    terminal_payoffs: list[Chips | None] = [None]
    node_states = [state]
    actions_by_node: list[tuple[Action, ...]] = [root_actions]
    children: list[ChildLink] = []

    for action in root_actions:
        child_state = apply_action(state, action)
        child_node_index = len(node_types)
        children.append(ChildLink(child=NodeId(child_node_index)))
        node_states.append(child_state)
        actions_by_node.append(())
        first_child.append(len(children))
        child_count.append(0)
        infoset_ids.append(None)

        if child_state.phase.value == "terminal":
            node_types.append(NodeType.TERMINAL)
            terminal_payoffs.append(Chips(0))
        else:
            node_types.append(NodeType.LEAF)
            terminal_payoffs.append(None)

    tree = PublicTree(
        node_types=tuple(node_types),
        first_child=tuple(first_child),
        child_count=tuple(child_count),
        children=tuple(children),
        infoset_ids=tuple(infoset_ids),
        terminal_payoffs=tuple(terminal_payoffs),
    )
    return BuiltPublicTree(
        tree=tree,
        node_states=tuple(node_states),
        actions_by_node=tuple(actions_by_node),
    )
