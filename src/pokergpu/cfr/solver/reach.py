from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Mapping

from pokergpu.tree.public_tree import InfosetId, NodeId, NodeType, PublicTree

from .infosets import DenseInfosetTable, build_dense_infoset_table
from .chunking import chunk_indices


@dataclass(slots=True, frozen=True)
class ReachResult:
    node_reach: tuple[float, ...]
    infoset_reach: tuple[float, ...]
    action_reach: tuple[tuple[float, ...], ...]
    cumulative_strategy: tuple[tuple[float, ...], ...]


def normalize_strategy(weights: tuple[float, ...]) -> tuple[float, ...]:
    assert weights, "strategy weights cannot be empty"
    if not weights:
        raise ValueError("strategy weights cannot be empty")
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0.0:
        return tuple(1.0 / len(weights) for _ in weights)
    return tuple(max(0.0, weight) / total for weight in weights)


def propagate_reach(
    tree: PublicTree,
    *,
    root_reach: float = 1.0,
    infoset_strategies: Mapping[InfosetId, tuple[float, ...]] | None = None,
    infoset_table: DenseInfosetTable | None = None,
    max_workers: int | None = None,
) -> ReachResult:
    table = infoset_table or build_dense_infoset_table(tree)
    assert len(table.node_to_infoset) == tree.node_count, "infoset table must align with tree"
    strategies = infoset_strategies or {}
    node_reach = [0.0 for _ in range(tree.node_count)]
    infoset_reach_map: dict[int, float] = {}
    cumulative_strategy_map: dict[int, list[float]] = {}
    action_reach: list[tuple[float, ...]] = [() for _ in range(tree.node_count)]

    node_reach[0] = root_reach
    assert node_reach[0] >= 0.0, "root reach must be non-negative"

    for node_index in range(tree.node_count):
        current_reach = node_reach[node_index]
        if current_reach <= 0.0:
            continue
        if tree.node_types[node_index] is NodeType.CHANCE:
            child_links = tree.child_links(NodeId(node_index))
            total_prob = sum(link.chance_prob or 0.0 for link in child_links)
            assert child_links, "chance nodes must have children"
            if abs(total_prob - 1.0) > 1e-6:
                raise ValueError("chance probabilities must sum to 1")
            for link in child_links:
                if link.chance_prob is None:
                    raise ValueError("chance children must define probabilities")
                node_reach[int(link.child)] += current_reach * link.chance_prob

    def process_infoset(
        infoset_id: int,
    ) -> tuple[int, float, tuple[float, ...], list[tuple[int, float]], list[tuple[int, tuple[float, ...]]]]:
        infoset_reach = 0.0
        cumulative: list[float] | None = None
        updates: list[tuple[int, float]] = []
        action_rows: list[tuple[int, tuple[float, ...]]] = []
        nodes = table.infoset_nodes[infoset_id]
        assert nodes, "infoset must have at least one node"
        local_node_reach = {node_index: node_reach[node_index] for node_index in nodes}
        for node_index in nodes:
            current_reach = local_node_reach[node_index]
            if current_reach <= 0.0:
                continue
            child_links = tree.child_links(NodeId(node_index))
            strategy = normalize_strategy(
                strategies.get(InfosetId(infoset_id), tuple(1.0 for _ in child_links))
            )
            if len(strategy) != len(child_links):
                raise ValueError("strategy length must match the node branching factor")
            if table.action_counts[infoset_id] != len(child_links):
                raise ValueError("infoset action count must match the node branching factor")

            infoset_reach += current_reach
            action_rows.append((node_index, strategy))
            if cumulative is None:
                cumulative = [0.0 for _ in range(len(strategy))]
            for child_index, link in enumerate(child_links):
                delta = current_reach * strategy[child_index]
                cumulative[child_index] += delta
                updates.append((int(link.child), delta))
                if int(link.child) in local_node_reach:
                    local_node_reach[int(link.child)] += delta
        return infoset_id, infoset_reach, tuple(cumulative or ()), updates, action_rows

    infoset_ids = list(table.infoset_order)
    if max_workers is None or max_workers <= 1 or len(infoset_ids) <= 1:
        results = [process_infoset(infoset_id) for infoset_id in infoset_ids]
    else:
        chunks = chunk_indices(infoset_ids, max_workers)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chunk_result in executor.map(lambda chunk: tuple(process_infoset(infoset_id) for infoset_id in chunk), chunks):
                results.extend(chunk_result)

    for infoset_id, reach, cumulative, updates, action_rows in results:
        infoset_reach_map[infoset_id] = infoset_reach_map.get(infoset_id, 0.0) + reach
        if cumulative:
            cumulative_strategy_map[infoset_id] = list(cumulative)
        for node_index, strategy in action_rows:
            action_reach[node_index] = strategy
        for child_index, delta in updates:
            node_reach[child_index] += delta

    max_infoset = max(infoset_reach_map, default=-1)
    infoset_reach = [0.0 for _ in range(max_infoset + 1)]
    cumulative_strategy: list[tuple[float, ...]] = [tuple() for _ in range(max_infoset + 1)]
    for infoset_id, reach in infoset_reach_map.items():
        infoset_reach[infoset_id] = reach
    for infoset_id, values in cumulative_strategy_map.items():
        cumulative_strategy[infoset_id] = tuple(values)

    for infoset_id, nodes in enumerate(table.infoset_nodes):
        if not nodes:
            continue
        if infoset_id >= len(infoset_reach):
            continue
        node_branching = tree.child_count[nodes[0]]
        for node_index in nodes[1:]:
            if tree.child_count[node_index] != node_branching:
                raise ValueError("infoset nodes must share the same branching factor")
            assert tree.node_types[node_index] in {NodeType.PLAYER0, NodeType.PLAYER1}

    return ReachResult(
        node_reach=tuple(node_reach),
        infoset_reach=tuple(infoset_reach),
        action_reach=tuple(action_reach),
        cumulative_strategy=tuple(cumulative_strategy),
    )
