from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class OpponentReachResult:
    infoset_opponent_reach: tuple[float, ...]
    infoset_card_opponent_reach: tuple[tuple[float, ...], ...]
    infoset_node_card_ratio: tuple[tuple[tuple[float, ...], ...], ...]
    node_opponent_reach: tuple[float, ...]
    node_opponent_share: tuple[float, ...]


def compute_opponent_reach(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    *,
    max_workers: int | None = None,
) -> OpponentReachResult:
    if tree.node_count != len(aggregate.node_aggregate.reach):
        raise ValueError("tree and aggregate result must cover the same number of nodes")

    infoset_nodes = _collect_infoset_nodes(tree)

    node_opponent_reach = tuple(aggregate.node_aggregate.reach)
    node_opponent_share = [0.0 for _ in range(tree.node_count)]
    infoset_opponent_reach = [0.0 for _ in range(len(infoset_nodes))]
    infoset_card_opponent_reach = [tuple(0.0 for _ in range(52)) for _ in range(len(infoset_nodes))]
    infoset_node_card_ratio: list[tuple[tuple[float, ...], ...]] = [tuple() for _ in range(len(infoset_nodes))]

    def process_infoset(
        infoset_id: int,
    ) -> tuple[int, float, tuple[float, ...], tuple[tuple[float, ...], ...], list[tuple[int, float]]]:
        nodes = infoset_nodes[infoset_id]
        if not nodes:
            raise ValueError("infoset must have at least one node")

        reach_total = 0.0
        card_reach = [0.0 for _ in range(52)]
        node_card_reach_rows: list[tuple[float, ...]] = []
        for node_index in nodes:
            if tree.node_types[node_index] not in {NodeType.PLAYER0, NodeType.PLAYER1}:
                raise ValueError("infoset nodes must be player nodes")
            reach_total += aggregate.node_aggregate.reach[node_index]
            node_card_reach = aggregate.node_aggregate.card_reach[node_index]
            if len(node_card_reach) != 52:
                raise ValueError("node card reach vectors must have length 52")
            for card_index, value in enumerate(node_card_reach):
                card_reach[card_index] += value
            node_card_reach_rows.append(node_card_reach)

        card_total = tuple(card_reach)
        node_card_ratio_rows: list[tuple[float, ...]] = []
        for node_card_reach in node_card_reach_rows:
            ratios = []
            for card_index, value in enumerate(node_card_reach):
                total = card_total[card_index]
                ratios.append(0.0 if total <= 0.0 else value / total)
            node_card_ratio_rows.append(tuple(ratios))

        node_shares: list[tuple[int, float]] = []
        if reach_total > 0.0:
            for node_index in nodes:
                share = aggregate.node_aggregate.reach[node_index] / reach_total
                node_shares.append((node_index, share))
        else:
            uniform_share = 1.0 / len(nodes)
            for node_index in nodes:
                node_shares.append((node_index, uniform_share))

        return infoset_id, reach_total, card_total, tuple(node_card_ratio_rows), node_shares

    infoset_ids = list(range(len(infoset_nodes)))
    if max_workers is None or max_workers <= 1 or len(infoset_ids) <= 1:
        results = [process_infoset(infoset_id) for infoset_id in infoset_ids]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_infoset, infoset_ids))

    for infoset_id, reach_total, card_reach, node_card_ratios, node_shares in results:
        infoset_opponent_reach[infoset_id] = reach_total
        infoset_card_opponent_reach[infoset_id] = card_reach
        infoset_node_card_ratio[infoset_id] = node_card_ratios
        for node_index, share in node_shares:
            node_opponent_share[node_index] = share

    return OpponentReachResult(
        infoset_opponent_reach=tuple(infoset_opponent_reach),
        infoset_card_opponent_reach=tuple(infoset_card_opponent_reach),
        infoset_node_card_ratio=tuple(infoset_node_card_ratio),
        node_opponent_reach=node_opponent_reach,
        node_opponent_share=tuple(node_opponent_share),
    )


def _collect_infoset_nodes(tree: PublicTree) -> tuple[tuple[int, ...], ...]:
    infoset_to_nodes: dict[int, list[int]] = {}
    for node_index, node_type in enumerate(tree.node_types):
        if node_type not in {NodeType.PLAYER0, NodeType.PLAYER1}:
            continue
        raw_infoset_id = tree.infoset_ids[node_index]
        if raw_infoset_id is None:
            raise ValueError("player nodes must have infoset ids")
        infoset_index = int(raw_infoset_id)
        infoset_to_nodes.setdefault(infoset_index, []).append(node_index)

    if not infoset_to_nodes:
        return ()

    max_infoset = max(infoset_to_nodes)
    dense_nodes: list[tuple[int, ...]] = [tuple() for _ in range(max_infoset + 1)]
    for infoset_id, nodes in infoset_to_nodes.items():
        dense_nodes[int(infoset_id)] = tuple(nodes)
    if any(not nodes for nodes in dense_nodes):
        raise ValueError("infoset ids must be dense and contiguous")
    return tuple(dense_nodes)
