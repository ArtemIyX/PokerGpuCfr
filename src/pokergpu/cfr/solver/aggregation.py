from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pokergpu.tree.public_tree import NodeId, NodeType, PublicTree


def aggregate_root_action_values(
    tree: PublicTree,
    *,
    max_workers: int | None = None,
) -> tuple[float, ...]:
    assert tree.node_count > 0, "public tree cannot be empty"
    root_children = tree.child_links(NodeId(0))
    if max_workers is None or max_workers <= 1 or len(root_children) <= 1:
        return _child_action_values(tree, NodeId(0))

    def evaluate_child(link_index: int) -> float:
        link = root_children[link_index]
        child_index = int(link.child)
        return _subtree_value(tree, NodeId(child_index))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return tuple(executor.map(evaluate_child, range(len(root_children))))


def aggregate_action_values(tree: PublicTree, node: NodeId) -> tuple[float, ...]:
    assert int(node) >= 0, "node id must be non-negative"
    return _child_action_values(tree, node)


def _child_action_values(tree: PublicTree, node: NodeId) -> tuple[float, ...]:
    return tuple(
        _subtree_value(tree, NodeId(int(link.child)))
        for link in tree.child_links(node)
    )


def _subtree_value(tree: PublicTree, node: NodeId) -> float:
    node_index = int(node)
    node_type = tree.node_types[node_index]
    if node_type is NodeType.TERMINAL:
        payoff = tree.terminal_payoffs[node_index]
        if payoff is None:
            raise ValueError("terminal nodes must carry payoffs")
        return float(payoff)
    if node_type is NodeType.LEAF:
        raise ValueError("toy solver cannot evaluate leaf nodes yet")

    postorder: list[int] = []
    stack: list[tuple[int, bool]] = [(node_index, False)]

    while stack:
        current, expanded = stack.pop()
        if expanded:
            postorder.append(current)
            continue

        stack.append((current, True))
        for link in tree.child_links(NodeId(current)):
            child_index = int(link.child)
            assert 0 <= child_index < tree.node_count, "child id must be in bounds"
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

        node_type = tree.node_types[current]
        if node_type is NodeType.CHANCE:
            if not child_values:
                raise ValueError("non-terminal nodes must have at least one value")
            total = 0.0
            for link in tree.child_links(NodeId(current)):
                if link.chance_prob is None:
                    raise ValueError("chance children must define probabilities")
                child_index = int(link.child)
                child_type = tree.node_types[child_index]
                payoff = tree.terminal_payoffs[child_index]
                if child_type is NodeType.TERMINAL:
                    if payoff is None:
                        raise ValueError("terminal nodes must carry payoffs")
                    contribution = float(payoff)
                else:
                    contribution = node_values[child_index]
                total += link.chance_prob * contribution
            node_values[current] = total
        else:
            if not child_values:
                raise ValueError("non-terminal nodes must have at least one value")
            node_values[current] = sum(child_values) / len(child_values)
    return node_values[node_index]
