from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from pokergpu.tree.public_tree import InfosetId, NodeId, NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class ForwardProfileResult:
    node_reach: tuple[float, ...]
    infoset_reach: tuple[float, ...]
    action_reach: tuple[tuple[float, ...], ...]


def normalize_strategy(weights: tuple[float, ...]) -> tuple[float, ...]:
    if not weights:
        return ()
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0.0:
        return tuple(1.0 / len(weights) for _ in weights)
    return tuple(max(0.0, weight) / total for weight in weights)


def propagate_forward(
    tree: PublicTree,
    *,
    root_reach: float = 1.0,
    infoset_strategies: Mapping[InfosetId, tuple[float, ...]] | None = None,
) -> ForwardProfileResult:
    strategies = infoset_strategies or {}
    node_reach = [0.0 for _ in range(tree.node_count)]
    infoset_ids = [int(info) if info is not None else -1 for info in tree.infoset_ids]
    infoset_reach_map: dict[int, float] = {}
    action_reach: list[tuple[float, ...]] = [() for _ in range(tree.node_count)]

    node_reach[0] = root_reach

    for node_index in range(tree.node_count):
        current_reach = node_reach[node_index]
        if current_reach <= 0.0:
            continue

        node_type = tree.node_types[node_index]
        if node_type is NodeType.CHANCE:
            child_links = tree.child_links(NodeId(node_index))
            if not child_links:
                raise ValueError("chance nodes must have children")
            for link in child_links:
                if link.chance_prob is None:
                    raise ValueError("chance children must define probabilities")
                node_reach[int(link.child)] += current_reach * float(link.chance_prob)
            continue

        if node_type not in {NodeType.PLAYER0, NodeType.PLAYER1}:
            continue

        infoset_id = infoset_ids[node_index]
        if infoset_id < 0:
            continue

        child_links = tree.child_links(NodeId(node_index))
        if not child_links:
            continue
        strategy = _aligned_strategy(
            strategies.get(InfosetId(infoset_id)),
            branching_factor=len(child_links),
        )

        infoset_reach_map[infoset_id] = infoset_reach_map.get(infoset_id, 0.0) + current_reach
        action_reach[node_index] = strategy

        for child_index, link in enumerate(child_links):
            node_reach[int(link.child)] += current_reach * strategy[child_index]

    max_infoset = max(infoset_reach_map, default=-1)
    infoset_reach = [0.0 for _ in range(max_infoset + 1)]
    for infoset_id, reach in infoset_reach_map.items():
        infoset_reach[infoset_id] = reach

    return ForwardProfileResult(
        node_reach=tuple(node_reach),
        infoset_reach=tuple(infoset_reach),
        action_reach=tuple(action_reach),
    )


def _aligned_strategy(
    strategy: tuple[float, ...] | None,
    *,
    branching_factor: int,
) -> tuple[float, ...]:
    if branching_factor <= 0:
        return ()
    if strategy is None or len(strategy) != branching_factor:
        return tuple(1.0 / branching_factor for _ in range(branching_factor))
    return normalize_strategy(strategy)
