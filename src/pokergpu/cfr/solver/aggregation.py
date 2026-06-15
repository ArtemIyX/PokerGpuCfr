from __future__ import annotations

from pokergpu.tree.public_tree import NodeId, NodeType, PublicTree


def aggregate_root_action_values(tree: PublicTree) -> tuple[float, ...]:
    return _child_action_values(tree, NodeId(0))


def aggregate_action_values(tree: PublicTree, node: NodeId) -> tuple[float, ...]:
    return _child_action_values(tree, node)


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
        node_type = tree.node_types[current]
        if node_type is NodeType.CHANCE:
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
