from __future__ import annotations

from dataclasses import dataclass

from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class DenseInfosetTable:
    node_to_infoset: tuple[int, ...]
    infoset_to_node: tuple[int, ...]
    action_counts: tuple[int, ...]
    infoset_nodes: tuple[tuple[int, ...], ...]

    @property
    def infoset_count(self) -> int:
        return len(self.infoset_to_node)


def build_dense_infoset_table(tree: PublicTree) -> DenseInfosetTable:
    assert tree.node_count > 0, "public tree cannot be empty"
    node_to_infoset = [-1 for _ in range(tree.node_count)]
    infoset_to_node: dict[int, int] = {}
    action_counts: dict[int, int] = {}
    infoset_nodes: dict[int, list[int]] = {}

    for node_index, node_type in enumerate(tree.node_types):
        if node_type not in {NodeType.PLAYER0, NodeType.PLAYER1}:
            continue
        infoset_id = tree.infoset_ids[node_index]
        if infoset_id is None:
            raise ValueError("player nodes must have infoset ids")
        dense_id = int(infoset_id)
        node_to_infoset[node_index] = dense_id
        infoset_to_node.setdefault(dense_id, node_index)
        action_counts[dense_id] = tree.child_count[node_index]
        infoset_nodes.setdefault(dense_id, []).append(node_index)
        assert tree.child_count[node_index] > 0, "player infosets must have actions"

    if infoset_to_node:
        max_infoset = max(infoset_to_node)
        dense_infoset_to_node = [-1 for _ in range(max_infoset + 1)]
        dense_action_counts = [0 for _ in range(max_infoset + 1)]
        for dense_id, node_index in infoset_to_node.items():
            dense_infoset_to_node[dense_id] = node_index
            dense_action_counts[dense_id] = action_counts[dense_id]
    else:
        dense_infoset_to_node = []
        dense_action_counts = []

    return DenseInfosetTable(
        node_to_infoset=tuple(node_to_infoset),
        infoset_to_node=tuple(dense_infoset_to_node),
        action_counts=tuple(dense_action_counts),
        infoset_nodes=tuple(tuple(nodes) for _, nodes in sorted(infoset_nodes.items())),
    )
