from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class DenseInfosetTable:
    node_to_infoset: tuple[int, ...]
    infoset_to_node: tuple[int, ...]
    action_counts: tuple[int, ...]
    action_labels: tuple[tuple[str, ...], ...]
    infoset_nodes: tuple[tuple[int, ...], ...]
    infoset_order: tuple[int, ...]
    infoset_node_counts: tuple[int, ...]

    @property
    def infoset_count(self) -> int:
        return len(self.infoset_to_node)


@lru_cache(maxsize=256)
def build_dense_infoset_table(tree: PublicTree) -> DenseInfosetTable:
    assert tree.node_count > 0, "public tree cannot be empty"
    node_to_infoset = [-1 for _ in range(tree.node_count)]
    infoset_to_node: dict[int, int] = {}
    action_counts: dict[int, int] = {}
    action_labels: dict[int, tuple[str, ...]] = {}
    infoset_nodes: dict[int, list[int]] = {}
    sparse_to_dense: dict[int, int] = {}

    for node_index, node_type in enumerate(tree.node_types):
        if node_type not in {NodeType.PLAYER0, NodeType.PLAYER1}:
            continue
        infoset_id = tree.infoset_ids[node_index]
        if infoset_id is None:
            raise ValueError("player nodes must have infoset ids")
        if tree.child_count[node_index] <= 0:
            continue
        sparse_id = int(infoset_id)
        dense_id = sparse_to_dense.setdefault(sparse_id, len(sparse_to_dense))
        node_to_infoset[node_index] = dense_id
        infoset_to_node.setdefault(dense_id, node_index)
        child_count = tree.child_count[node_index]
        previous_count = action_counts.get(dense_id)
        if previous_count is None:
            action_counts[dense_id] = child_count
        elif previous_count != child_count:
            raise ValueError("infoset nodes must share the same branching factor")
        labels = tree.action_labels[node_index]
        if labels is None:
            labels = tuple(f"action_{index}" for index in range(child_count))
        elif len(labels) != child_count:
            labels = tuple(f"action_{index}" for index in range(child_count))
        previous_labels = action_labels.get(dense_id)
        if previous_labels is None:
            action_labels[dense_id] = labels
        elif previous_labels != labels:
            raise ValueError("infoset nodes must share identical action labels")
        infoset_nodes.setdefault(dense_id, []).append(node_index)

    if infoset_to_node:
        dense_count = len(infoset_to_node)
        dense_infoset_to_node = [-1 for _ in range(dense_count)]
        dense_action_counts = [0 for _ in range(dense_count)]
        dense_action_labels: list[tuple[str, ...]] = [() for _ in range(dense_count)]
        for dense_id, node_index in infoset_to_node.items():
            dense_infoset_to_node[dense_id] = node_index
            dense_action_counts[dense_id] = action_counts[dense_id]
            dense_action_labels[dense_id] = action_labels[dense_id]
    else:
        dense_infoset_to_node = []
        dense_action_counts = []
        dense_action_labels = []

    ordered_infosets = sorted(infoset_nodes.items())
    return DenseInfosetTable(
        node_to_infoset=tuple(node_to_infoset),
        infoset_to_node=tuple(dense_infoset_to_node),
        action_counts=tuple(dense_action_counts),
        action_labels=tuple(dense_action_labels),
        infoset_nodes=tuple(tuple(nodes) for _, nodes in ordered_infosets),
        infoset_order=tuple(infoset_id for infoset_id, _ in ordered_infosets),
        infoset_node_counts=tuple(len(nodes) for _, nodes in ordered_infosets),
    )
