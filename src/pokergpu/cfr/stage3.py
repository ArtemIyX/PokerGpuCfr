from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Sequence

from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.abstraction.hands import private_hand_count
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True, frozen=True)
class OpponentReachResult:
    """Blocking-aware opponent reach summaries for a single public tree."""

    infoset_opponent_reach: tuple[float, ...]
    infoset_card_opponent_reach: tuple[tuple[float, ...], ...]
    infoset_hand_opponent_reach: tuple[tuple[float, ...], ...]
    infoset_node_hand_ratio: tuple[tuple[tuple[float, ...], ...], ...]
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
    if not aggregate.node_aggregate.hand_reach:
        raise ValueError("node hand reach vectors cannot be empty")

    infoset_nodes = _collect_infoset_nodes(tree)
    hand_width = len(aggregate.node_aggregate.hand_reach[0])
    if hand_width != private_hand_count():
        raise ValueError("node hand reach vectors must match the private hand count")

    node_opponent_reach = tuple(aggregate.node_aggregate.reach)
    node_opponent_share = [0.0 for _ in range(tree.node_count)]
    infoset_opponent_reach = [0.0 for _ in range(len(infoset_nodes))]
    infoset_card_opponent_reach = [tuple(0.0 for _ in range(52)) for _ in range(len(infoset_nodes))]
    infoset_hand_opponent_reach = [tuple(0.0 for _ in range(hand_width)) for _ in range(len(infoset_nodes))]
    infoset_node_hand_ratio: list[tuple[tuple[float, ...], ...]] = [tuple() for _ in range(len(infoset_nodes))]
    infoset_node_card_ratio: list[tuple[tuple[float, ...], ...]] = [tuple() for _ in range(len(infoset_nodes))]

    def process_infoset(
        infoset_id: int,
    ) -> tuple[
        int,
        float,
        tuple[float, ...],
        tuple[float, ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
        list[tuple[int, float]],
    ]:
        nodes = infoset_nodes[infoset_id]
        if not nodes:
            raise ValueError("infoset must have at least one node")

        reach_total = 0.0
        card_reach = [0.0 for _ in range(52)]
        hand_reach = [0.0 for _ in range(hand_width)]
        node_card_reach_rows: list[tuple[float, ...]] = []
        node_hand_reach_rows: list[tuple[float, ...]] = []
        for node_index in nodes:
            if tree.node_types[node_index] not in {NodeType.PLAYER0, NodeType.PLAYER1}:
                raise ValueError("infoset nodes must be player nodes")
            reach_total += aggregate.node_aggregate.reach[node_index]
            node_card_reach = aggregate.node_aggregate.card_reach[node_index]
            node_hand_reach = aggregate.node_aggregate.hand_reach[node_index]
            if len(node_card_reach) != 52:
                raise ValueError("node card reach vectors must have length 52")
            if len(node_hand_reach) != hand_width:
                raise ValueError("node hand reach vectors must have consistent length")
            for card_index, value in enumerate(node_card_reach):
                card_reach[card_index] += value
            for hand_index, value in enumerate(node_hand_reach):
                hand_reach[hand_index] += value
            node_card_reach_rows.append(node_card_reach)
            node_hand_reach_rows.append(node_hand_reach)

        card_total = tuple(card_reach)
        hand_total = tuple(hand_reach)
        node_card_ratio_rows = _normalize_card_reach_rows(node_card_reach_rows, card_total)
        node_hand_ratio_rows = _normalize_blocking_reach_rows(node_hand_reach_rows, hand_total)
        _validate_card_ratio_rows(node_card_ratio_rows)
        _validate_hand_totals(node_hand_reach_rows, hand_total)
        _validate_hand_ratio_rows(node_hand_ratio_rows, hand_width)

        node_shares: list[tuple[int, float]] = []
        if reach_total > 0.0:
            for node_index in nodes:
                share = aggregate.node_aggregate.reach[node_index] / reach_total
                node_shares.append((node_index, share))
        else:
            uniform_share = 1.0 / len(nodes)
            for node_index in nodes:
                node_shares.append((node_index, uniform_share))

        return (
            infoset_id,
            reach_total,
            card_total,
            hand_total,
            tuple(node_hand_ratio_rows),
            tuple(node_card_ratio_rows),
            node_shares,
        )

    infoset_ids = list(range(len(infoset_nodes)))
    if max_workers is None or max_workers <= 1 or len(infoset_ids) <= 1:
        results = [process_infoset(infoset_id) for infoset_id in infoset_ids]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_infoset, infoset_ids))

    for infoset_id, reach_total, card_reach, hand_reach, node_hand_ratios, node_card_ratios, node_shares in results:
        infoset_opponent_reach[infoset_id] = reach_total
        infoset_card_opponent_reach[infoset_id] = card_reach
        infoset_hand_opponent_reach[infoset_id] = hand_reach
        infoset_node_hand_ratio[infoset_id] = node_hand_ratios
        infoset_node_card_ratio[infoset_id] = node_card_ratios
        for node_index, share in node_shares:
            node_opponent_share[node_index] = share

    return OpponentReachResult(
        infoset_opponent_reach=tuple(infoset_opponent_reach),
        infoset_card_opponent_reach=tuple(infoset_card_opponent_reach),
        infoset_hand_opponent_reach=tuple(infoset_hand_opponent_reach),
        infoset_node_hand_ratio=tuple(infoset_node_hand_ratio),
        infoset_node_card_ratio=tuple(infoset_node_card_ratio),
        node_opponent_reach=node_opponent_reach,
        node_opponent_share=tuple(node_opponent_share),
    )


def _normalize_card_reach_rows(
    node_card_reach_rows: list[tuple[float, ...]],
    card_total: tuple[float, ...],
) -> list[tuple[float, ...]]:
    normalized_rows: list[tuple[float, ...]] = []
    for node_card_reach in node_card_reach_rows:
        ratios: list[float] = []
        for card_index, value in enumerate(node_card_reach):
            total = card_total[card_index]
            ratios.append(0.0 if total <= 0.0 else value / total)
        normalized_rows.append(tuple(ratios))
    return normalized_rows


def _normalize_blocking_reach_rows(
    node_reach_rows: list[tuple[float, ...]],
    total_reach: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    if not node_reach_rows:
        return ()
    normalized_rows: list[tuple[float, ...]] = []
    for row in node_reach_rows:
        if len(row) != len(total_reach):
            raise ValueError("reach rows must have consistent width")
        ratios: list[float] = []
        for index, value in enumerate(row):
            total = total_reach[index]
            ratios.append(0.0 if total <= 0.0 else value / total)
        normalized_rows.append(tuple(ratios))
    return tuple(normalized_rows)


def _validate_card_ratio_rows(node_card_ratio_rows: list[tuple[float, ...]]) -> None:
    if not node_card_ratio_rows:
        return
    card_width = len(node_card_ratio_rows[0])
    if card_width != 52:
        raise ValueError("card ratio vectors must have length 52")
    for row in node_card_ratio_rows[1:]:
        if len(row) != card_width:
            raise ValueError("card ratio rows must have consistent width")


def _validate_hand_totals(
    node_hand_reach_rows: list[tuple[float, ...]],
    hand_total: tuple[float, ...],
) -> None:
    if not node_hand_reach_rows:
        return
    hand_width = len(hand_total)
    if any(len(row) != hand_width for row in node_hand_reach_rows):
        raise ValueError("hand reach rows must have consistent width")


def _validate_hand_ratio_rows(
    node_hand_ratio_rows: Sequence[tuple[float, ...]],
    hand_width: int,
) -> None:
    for row in node_hand_ratio_rows:
        if len(row) != hand_width:
            raise ValueError("hand ratio rows must have consistent width")


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
