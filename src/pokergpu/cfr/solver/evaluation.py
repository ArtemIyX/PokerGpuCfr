from __future__ import annotations

from pokergpu.tree.public_tree import PublicTree

from .aggregation import aggregate_root_action_values


def evaluate_root_action_values(tree: PublicTree) -> tuple[float, ...]:
    return aggregate_root_action_values(tree)
