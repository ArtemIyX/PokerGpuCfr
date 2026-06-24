from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from pokergpu.abstraction.actions import ActionAbstraction, BaselineActionAbstraction
from pokergpu.core.actions import Action
from pokergpu.core.betting import Chips
from pokergpu.core.state import GameState, HandPhase
from pokergpu.core.transitions import advance_hand_to_next_street
from pokergpu.core.transitions import apply_action
from pokergpu.core.board import Board

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


@dataclass(slots=True, frozen=True)
class ChanceOutcome:
    state: GameState
    probability: float


def build_shallow_public_tree(
    state: GameState,
    *,
    abstraction: ActionAbstraction | None = None,
    advance_next_board: Callable[[GameState], Board | None] | None = None,
    expand_chance: Callable[[GameState], tuple[ChanceOutcome, ...] | None] | None = None,
) -> BuiltPublicTree:
    return build_public_tree(
        state,
        abstraction=abstraction,
        config=TreeBuildConfig(max_depth=1),
        advance_next_board=advance_next_board,
        expand_chance=expand_chance,
    )


def build_public_tree(
    state: GameState,
    *,
    abstraction: ActionAbstraction | None = None,
    config: TreeBuildConfig | None = None,
    advance_next_board: Callable[[GameState], Board | None] | None = None,
    expand_chance: Callable[[GameState], tuple[ChanceOutcome, ...] | None] | None = None,
) -> BuiltPublicTree:
    abstraction_impl = abstraction or BaselineActionAbstraction()
    build_config = config or TreeBuildConfig()

    root_chance_outcomes = expand_chance(state) if expand_chance is not None else None
    root_is_chance = root_chance_outcomes is not None
    node_types: list[NodeType] = [
        NodeType.CHANCE if root_is_chance else _node_type_for_state(state, depth=0, max_depth=build_config.max_depth)
    ]
    first_child: list[int] = [0]
    child_count: list[int] = [0]
    infoset_ids: list[InfosetId | None] = []
    next_infoset_id = 0
    root_infoset = _infoset_id_for_state(state, node_types[0], next_infoset_id)
    if root_infoset is not None:
        next_infoset_id += 1
    infoset_ids.append(root_infoset)
    terminal_payoffs: list[Chips | None] = [_terminal_payoff_for_state(state)]
    node_states: list[GameState] = [state]
    actions_by_node: list[tuple[Action, ...]] = [()]
    children: list[ChildLink] = []

    queue: deque[tuple[int, int]] = deque([(0, 0)])

    if root_chance_outcomes is not None:
        actions_by_node[0] = ()
        first_child[0] = len(children)
        child_count[0] = 0
        added_children = 0
        for outcome in root_chance_outcomes:
            if len(node_states) >= build_config.max_nodes:
                break
            if outcome.probability < 0.0 or outcome.probability > 1.0:
                raise ValueError("chance outcome probability must be in [0, 1]")
            child_state = outcome.state
            child_node_index = len(node_states)
            children.append(ChildLink(child=NodeId(child_node_index), chance_prob=outcome.probability))
            node_states.append(child_state)
            actions_by_node.append(())
            child_type = _node_type_for_state(
                child_state,
                depth=1,
                max_depth=build_config.max_depth,
            )
            added_children += 1
            node_types.append(child_type)
            first_child.append(0)
            child_count.append(0)
            child_infoset = _infoset_id_for_state(child_state, child_type, next_infoset_id)
            if child_infoset is not None:
                next_infoset_id += 1
            infoset_ids.append(child_infoset)
            terminal_payoffs.append(_terminal_payoff_for_state(child_state))
            if child_type not in {NodeType.LEAF, NodeType.TERMINAL}:
                queue.append((child_node_index, 1))
        child_count[0] = added_children

    while queue and len(node_states) < build_config.max_nodes:
        node_index, depth = queue.popleft()
        node_state = node_states[node_index]
        node_type = node_types[node_index]

        if node_type in {NodeType.LEAF, NodeType.TERMINAL}:
            continue

        chance_outcomes = None
        if expand_chance is not None and not (node_index == 0 and root_is_chance):
            chance_outcomes = expand_chance(node_state)
        if chance_outcomes is not None:
            actions_by_node[node_index] = ()
            node_types[node_index] = NodeType.CHANCE
            first_child[node_index] = len(children)
            child_count[node_index] = 0
            added_children = 0
            for outcome in chance_outcomes:
                if len(node_states) >= build_config.max_nodes:
                    break
                if outcome.probability < 0.0 or outcome.probability > 1.0:
                    raise ValueError("chance outcome probability must be in [0, 1]")
                child_state = outcome.state
                child_node_index = len(node_states)
                children.append(ChildLink(child=NodeId(child_node_index), chance_prob=outcome.probability))
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
                child_infoset = _infoset_id_for_state(child_state, child_type, next_infoset_id)
                if child_infoset is not None:
                    next_infoset_id += 1
                infoset_ids.append(child_infoset)
                terminal_payoffs.append(_terminal_payoff_for_state(child_state))
                if child_type not in {NodeType.LEAF, NodeType.TERMINAL}:
                    queue.append((child_node_index, next_depth))
            child_count[node_index] = added_children
            continue

        if node_type is NodeType.CHANCE:
            continue

        legal_actions = abstraction_impl.legal_actions(node_state)
        if not legal_actions:
            node_types[node_index] = NodeType.LEAF
            actions_by_node[node_index] = ()
            child_count[node_index] = 0
            terminal_payoffs[node_index] = _terminal_payoff_for_state(node_state)
            continue
        actions_by_node[node_index] = legal_actions
        first_child[node_index] = len(children)
        added_children = 0

        for action in legal_actions:
            if len(node_states) >= build_config.max_nodes:
                break

            child_state = apply_action(node_state, action)
            child_state = _advance_closed_street(child_state, advance_next_board)
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
            child_infoset = _infoset_id_for_state(child_state, child_type, next_infoset_id)
            if child_infoset is not None:
                next_infoset_id += 1
            infoset_ids.append(child_infoset)
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
    node_type: NodeType,
    dense_infoset_id: int,
) -> InfosetId | None:
    if (
        node_type in {NodeType.PLAYER0, NodeType.PLAYER1}
        and state.phase is HandPhase.IN_PROGRESS
    ):
        return InfosetId(dense_infoset_id)
    return None


def _terminal_payoff_for_state(state: GameState) -> Chips | None:
    if state.phase is HandPhase.TERMINAL:
        return Chips(0)
    return None


def _advance_closed_street(
    state: GameState,
    advance_next_board: Callable[[GameState], Board | None] | None,
) -> GameState:
    if advance_next_board is None:
        return state
    if state.phase is not HandPhase.IN_PROGRESS:
        return state
    if state.board.is_river:
        return state
    if state.betting_round.to_act != state.dealer:
        return state
    next_board = advance_next_board(state)
    if next_board is None:
        return state
    next_cards = next_board.cards
    return advance_hand_to_next_street(state, next_board_cards=next_cards)
