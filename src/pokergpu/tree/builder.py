from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pokergpu.abstraction.actions import ActionAbstraction, BaselineActionAbstraction
from pokergpu.core.actions import Action
from pokergpu.core.betting import Chips
from pokergpu.core.state import GameState, HandPhase
from pokergpu.core.transitions import apply_action

from .public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class BuiltPublicTree:
    tree: PublicTree
    node_states: tuple[GameState, ...]
    actions_by_node: tuple[tuple[Action, ...], ...]


@dataclass(slots=True, frozen=True)
class TreeBuildConfig:
    max_depth: int = 1
    max_nodes: int = 256


def build_shallow_public_tree(
    state: GameState,
    *,
    abstraction: ActionAbstraction | None = None,
) -> BuiltPublicTree:
    return build_public_tree(
        state,
        abstraction=abstraction,
        config=TreeBuildConfig(max_depth=1),
    )


def build_public_tree(
    state: GameState,
    *,
    abstraction: ActionAbstraction | None = None,
    config: TreeBuildConfig | None = None,
) -> BuiltPublicTree:
    abstraction_impl = abstraction or BaselineActionAbstraction()
    build_config = config or TreeBuildConfig()

    node_types: list[NodeType] = [
        _node_type_for_state(state, depth=0, max_depth=build_config.max_depth)
    ]
    first_child: list[int] = [0]
    child_count: list[int] = [0]
    infoset_ids: list[InfosetId | None] = [
        _infoset_id_for_state(state, 0, node_types[0])
    ]
    terminal_payoffs: list[Chips | None] = [_terminal_payoff_for_state(state)]
    node_states: list[GameState] = [state]
    actions_by_node: list[tuple[Action, ...]] = [()]
    children: list[ChildLink] = []

    queue: deque[tuple[int, int]] = deque([(0, 0)])

    while queue and len(node_states) < build_config.max_nodes:
        node_index, depth = queue.popleft()
        node_state = node_states[node_index]
        node_type = node_types[node_index]

        if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
            continue

        legal_actions = abstraction_impl.legal_actions(node_state)
        actions_by_node[node_index] = legal_actions
        first_child[node_index] = len(children)
        added_children = 0

        for action in legal_actions:
            if len(node_states) >= build_config.max_nodes:
                break

            child_state = apply_action(node_state, action)
            child_node_index = len(node_states)
            children.append(ChildLink(child=NodeId(child_node_index)))
            node_states.append(child_state)
            actions_by_node.append(())
            next_depth = depth + 1
            child_type = _node_type_for_state(
                child_state,
                depth=next_depth,
                max_depth=build_config.max_depth,
            )
            added_children += 1
            node_types.append(child_type)
            first_child.append(0)
            child_count.append(0)
            infoset_ids.append(
                _infoset_id_for_state(child_state, child_node_index, child_type)
            )
            terminal_payoffs.append(_terminal_payoff_for_state(child_state))

            if child_type not in {NodeType.LEAF, NodeType.TERMINAL}:
                queue.append((child_node_index, next_depth))

        child_count[node_index] = added_children

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


def _node_type_for_state(state: GameState, *, depth: int, max_depth: int) -> NodeType:
    if state.phase is HandPhase.TERMINAL:
        return NodeType.TERMINAL
    if depth >= max_depth or state.phase is HandPhase.SHOWDOWN:
        return NodeType.LEAF
    return NodeType.PLAYER0 if state.betting_round.to_act == 0 else NodeType.PLAYER1


def _infoset_id_for_state(
    state: GameState,
    node_index: int,
    node_type: NodeType,
) -> InfosetId | None:
    if (
        node_type in {NodeType.PLAYER0, NodeType.PLAYER1}
        and state.phase is HandPhase.IN_PROGRESS
    ):
        return InfosetId(node_index)
    return None


def _terminal_payoff_for_state(state: GameState) -> Chips | None:
    if state.phase is HandPhase.TERMINAL:
        return Chips(0)
    return None
