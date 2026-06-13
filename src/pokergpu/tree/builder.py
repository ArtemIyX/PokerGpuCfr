from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pokergpu.abstraction.actions import ActionAbstraction, BaselineActionAbstraction
from pokergpu.core.actions import Action
from pokergpu.core.betting import Chips
from pokergpu.core.canonical import canonical_board_key
from pokergpu.core.payouts import compute_payouts
from pokergpu.core.state import GameState, HandPhase
from pokergpu.core.transitions import apply_action

from .public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class BuiltPublicTree:
    tree: PublicTree
    node_states: tuple[GameState, ...]
    actions_by_node: tuple[tuple[Action, ...], ...]
    action_abstraction_id: str
    canonical_board_key: str
    player_count: int
    active_players: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class TreeBuildConfig:
    max_depth: int = 1
    max_nodes: int = 256
    min_reach_prob: float = 0.0


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
    is_frontier: list[bool] = [
        _is_frontier_node(state, depth=0, max_depth=build_config.max_depth)
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
        if is_frontier[node_index]:
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
            child_is_frontier = _is_frontier_node(
                child_state,
                depth=next_depth,
                max_depth=build_config.max_depth,
            )
            added_children += 1
            node_types.append(child_type)
            is_frontier.append(child_is_frontier)
            first_child.append(0)
            child_count.append(0)
            infoset_ids.append(
                _infoset_id_for_state(child_state, child_node_index, child_type)
            )
            terminal_payoffs.append(_terminal_payoff_for_state(child_state))

            if (child_type not in {NodeType.LEAF, NodeType.TERMINAL}
                and not child_is_frontier):
                queue.append((child_node_index, next_depth))

        child_count[node_index] = added_children

    tree = PublicTree(
        node_types=tuple(node_types),
        is_frontier=tuple(is_frontier),
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
        action_abstraction_id=abstraction_impl.abstraction_id(state),
        canonical_board_key=canonical_board_key(state.board),
        player_count=state.player_count,
        active_players=tuple(int(player.player) for player in state.active_players),
    )


def _node_type_for_state(state: GameState, *, depth: int, max_depth: int) -> NodeType:
    if state.phase is HandPhase.TERMINAL:
        return NodeType.TERMINAL
    if depth >= max_depth or state.phase is HandPhase.SHOWDOWN:
        return NodeType.LEAF
    return _node_type_for_player(int(state.betting_round.to_act))


def _is_frontier_node(state: GameState, *, depth: int, max_depth: int) -> bool:
    return (
        state.phase is HandPhase.TERMINAL
        or depth >= max_depth
        or state.phase is HandPhase.SHOWDOWN
    )


def _infoset_id_for_state(
    state: GameState,
    node_index: int,
    node_type: NodeType,
) -> InfosetId | None:
    if (
        node_type
        in {
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.PLAYER2,
            NodeType.PLAYER3,
            NodeType.PLAYER4,
            NodeType.PLAYER5,
        }
        and state.phase is HandPhase.IN_PROGRESS
    ):
        return InfosetId(node_index)
    return None


def _terminal_payoff_for_state(state: GameState) -> Chips | None:
    if state.phase is HandPhase.TERMINAL:
        payouts = compute_payouts(state)
        player0_payout = next(
            (payout.amount for payout in payouts if payout.player == 0),
            Chips(0),
        )
        other_payouts = sum(
            payout.amount for payout in payouts if payout.player != 0
        )
        return Chips(player0_payout - other_payouts)
    return None


def _node_type_for_player(player: int) -> NodeType:
    if player == 0:
        return NodeType.PLAYER0
    if player == 1:
        return NodeType.PLAYER1
    if player == 2:
        return NodeType.PLAYER2
    if player == 3:
        return NodeType.PLAYER3
    if player == 4:
        return NodeType.PLAYER4
    if player == 5:
        return NodeType.PLAYER5
    raise ValueError("unsupported player seat")
