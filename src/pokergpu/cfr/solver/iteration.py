from __future__ import annotations

from pokergpu.tree.public_tree import NodeId, NodeType, PublicTree

from .state import ToyCfrResult, ToyCfrState
from ..stage1 import normalize_strategy
from ..stage7 import regret_matching, update_average_strategy, update_regret


def run_toy_cfr_iteration(
    state: ToyCfrState,
    action_values: tuple[float, ...],
    *,
    reach_weight: float = 1.0,
) -> ToyCfrResult:
    if not action_values:
        raise ValueError("action values cannot be empty")
    if len(state.regret_sums) != len(action_values):
        raise ValueError("state and action values must have the same length")
    if len(state.strategy_sums) != len(action_values):
        raise ValueError("state and action values must have the same length")

    strategy = normalize_strategy(regret_matching(state.regret_sums))
    node_value = sum(prob * value for prob, value in zip(strategy, action_values, strict=True))
    regret_sums = update_regret(state.regret_sums, action_values, node_value)
    strategy_sums = update_average_strategy(state.strategy_sums, strategy, reach_weight)

    return ToyCfrResult(
        state=ToyCfrState(regret_sums=regret_sums, strategy_sums=strategy_sums),
        strategy=strategy,
        node_value=node_value,
        action_values=action_values,
    )


def run_tree_root_cfr_iteration(
    tree: PublicTree,
    state: ToyCfrState,
    reach_weight: float = 1.0,
) -> ToyCfrResult:
    terminal_values = _child_action_values(tree, NodeId(0))

    return run_toy_cfr_iteration(
        state,
        terminal_values,
        reach_weight=reach_weight,
    )


def _child_action_values(tree: PublicTree, node: NodeId) -> tuple[float, ...]:
    postorder: list[int] = []
    stack: list[tuple[int, bool]] = [(int(node), False)]

    while stack:
        current, expanded = stack.pop()
        if expanded:
            postorder.append(current)
            continue

        stack.append((current, True))
        for link in tree.child_links(NodeId(current)):
            child_index = int(link.child)
            child_type = tree.node_types[child_index]
            if child_type is NodeType.LEAF:
                raise ValueError("toy solver cannot evaluate leaf nodes yet")
            if child_type is NodeType.TERMINAL:
                continue
            stack.append((child_index, False))

    node_values: dict[int, float] = {}
    for current in postorder:
        child_values: list[float] = []
        for link in tree.child_links(NodeId(current)):
            child_index = int(link.child)
            child_type = tree.node_types[child_index]
            payoff = tree.terminal_payoffs[child_index]

            if child_type is NodeType.TERMINAL:
                if payoff is None:
                    raise ValueError("terminal nodes must carry payoffs")
                child_values.append(float(payoff))
            else:
                child_values.append(node_values[child_index])

        if not child_values:
            raise ValueError("non-terminal nodes must have at least one value")
        node_values[current] = sum(child_values) / len(child_values)

    root_child_values: list[float] = []
    for link in tree.child_links(node):
        child_index = int(link.child)
        child_type = tree.node_types[child_index]
        payoff = tree.terminal_payoffs[child_index]
        if child_type is NodeType.TERMINAL:
            if payoff is None:
                raise ValueError("terminal nodes must carry payoffs")
            root_child_values.append(float(payoff))
        else:
            root_child_values.append(node_values[child_index])
    return tuple(root_child_values)
