from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.solver.infosets import DenseInfosetTable
from pokergpu.cfr.solver.infosets import build_dense_infoset_table
from pokergpu.abstraction.hands import private_hand_count
from pokergpu.tree.public_tree import PublicTree

try:
    from numba import njit, prange  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    njit = None
    prange = None


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
    node_hand_opponent_reach: tuple[tuple[float, ...], ...]


@dataclass(slots=True, frozen=True)
class Stage3PreparedInput:
    infoset_table: DenseInfosetTable
    infoset_node_indices: np.ndarray
    infoset_node_counts: np.ndarray
    node_reach: np.ndarray
    node_card_reach: np.ndarray
    node_hand_reach: np.ndarray


def compute_opponent_reach(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    *,
    max_workers: int | None = None,
) -> OpponentReachResult:
    if tree.node_count != len(aggregate.node_aggregate.reach):
        raise ValueError("tree and aggregate result must cover the same number of nodes")
    prepared = _prepare_stage3_input(tree, aggregate)
    infoset_table = prepared.infoset_table
    infoset_nodes = infoset_table.infoset_nodes
    hand_width = int(prepared.node_hand_reach.shape[1])
    if hand_width != private_hand_count():
        raise ValueError("node hand reach vectors must match the private hand count")

    node_reach = prepared.node_reach
    node_card_reach = prepared.node_card_reach
    node_hand_reach = prepared.node_hand_reach
    infoset_count = infoset_table.infoset_count

    infoset_opponent_reach = np.zeros(infoset_count, dtype=np.float64)
    infoset_card_opponent_reach = np.zeros((infoset_count, 52), dtype=np.float64)
    infoset_hand_opponent_reach = np.zeros((infoset_count, hand_width), dtype=np.float64)
    node_opponent_share = np.zeros(tree.node_count, dtype=np.float64)
    node_hand_opponent_reach = np.zeros((tree.node_count, hand_width), dtype=np.float64)
    infoset_node_hand_ratio = np.zeros((infoset_count, node_reach.shape[0], hand_width), dtype=np.float64)
    infoset_node_card_ratio = np.zeros((infoset_count, node_reach.shape[0], 52), dtype=np.float64)

    _reduce_stage3_infosets(
        infoset_node_indices=prepared.infoset_node_indices,
        infoset_node_counts=prepared.infoset_node_counts,
        node_reach=node_reach,
        node_card_reach=node_card_reach,
        node_hand_reach=node_hand_reach,
        infoset_opponent_reach=infoset_opponent_reach,
        infoset_card_opponent_reach=infoset_card_opponent_reach,
        infoset_hand_opponent_reach=infoset_hand_opponent_reach,
        infoset_node_hand_ratio=infoset_node_hand_ratio,
        infoset_node_card_ratio=infoset_node_card_ratio,
        node_opponent_share=node_opponent_share,
        node_hand_opponent_reach=node_hand_opponent_reach,
        max_workers=max_workers,
    )

    return OpponentReachResult(
        infoset_opponent_reach=tuple(float(value) for value in infoset_opponent_reach),
        infoset_card_opponent_reach=tuple(
            tuple(float(value) for value in row)
            for row in infoset_card_opponent_reach
        ),
        infoset_hand_opponent_reach=tuple(
            tuple(float(value) for value in row)
            for row in infoset_hand_opponent_reach
        ),
        infoset_node_hand_ratio=tuple(
            tuple(
                tuple(float(value) for value in node_row)
                for node_row in (
                    infoset_node_hand_ratio[infoset_id, node_index]
                    for node_index in range(int(prepared.infoset_node_counts[infoset_id]))
                )
            )
            for infoset_id in range(infoset_table.infoset_count)
        ),
        infoset_node_card_ratio=tuple(
            tuple(
                tuple(float(value) for value in node_row)
                for node_row in (
                    infoset_node_card_ratio[infoset_id, node_index]
                    for node_index in range(int(prepared.infoset_node_counts[infoset_id]))
                )
            )
            for infoset_id in range(infoset_table.infoset_count)
        ),
        node_opponent_reach=tuple(float(value) for value in node_reach),
        node_opponent_share=tuple(float(value) for value in node_opponent_share),
        node_hand_opponent_reach=tuple(tuple(float(value) for value in row) for row in node_hand_opponent_reach),
    )


def _prepare_stage3_input(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
) -> Stage3PreparedInput:
    """Cache-backed dense infoset layout and rectangular Stage 3 buffers."""
    infoset_table = build_dense_infoset_table(tree)
    node_reach = np.asarray(aggregate.node_aggregate.reach, dtype=np.float64)
    node_card_reach = np.asarray(aggregate.node_aggregate.card_reach, dtype=np.float64)
    node_hand_reach = np.asarray(aggregate.node_aggregate.hand_reach, dtype=np.float64)
    if node_card_reach.ndim != 2 or node_card_reach.shape[1] != 52:
        raise ValueError("node card reach vectors must have length 52")
    if node_hand_reach.ndim != 2:
        raise ValueError("node hand reach vectors must be rectangular")
    if node_reach.ndim != 1:
        raise ValueError("node reach must be one-dimensional")
    if node_card_reach.shape[0] != tree.node_count or node_hand_reach.shape[0] != tree.node_count:
        raise ValueError("tree and aggregate result must cover the same number of nodes")

    infoset_count = infoset_table.infoset_count
    max_infoset_width = max((len(nodes) for nodes in infoset_table.infoset_nodes), default=0)
    infoset_node_indices = np.full((infoset_count, max_infoset_width), -1, dtype=np.int64)
    infoset_node_counts = np.zeros(infoset_count, dtype=np.int64)
    for infoset_id, nodes in enumerate(infoset_table.infoset_nodes):
        count = len(nodes)
        infoset_node_counts[infoset_id] = count
        if count > 0:
            infoset_node_indices[infoset_id, :count] = np.asarray(nodes, dtype=np.int64)

    return Stage3PreparedInput(
        infoset_table=infoset_table,
        infoset_node_indices=infoset_node_indices,
        infoset_node_counts=infoset_node_counts,
        node_reach=node_reach,
        node_card_reach=node_card_reach,
        node_hand_reach=node_hand_reach,
    )


def _reduce_stage3_infosets(
    *,
    infoset_node_indices: np.ndarray,
    infoset_node_counts: np.ndarray,
    node_reach: np.ndarray,
    node_card_reach: np.ndarray,
    node_hand_reach: np.ndarray,
    infoset_opponent_reach: np.ndarray,
    infoset_card_opponent_reach: np.ndarray,
    infoset_hand_opponent_reach: np.ndarray,
    infoset_node_hand_ratio: np.ndarray,
    infoset_node_card_ratio: np.ndarray,
    node_opponent_share: np.ndarray,
    node_hand_opponent_reach: np.ndarray,
    max_workers: int | None,
) -> None:
    infoset_count = infoset_node_counts.shape[0]
    if infoset_count == 0:
        return
    if _can_use_numba_parallel(max_workers):
        _reduce_stage3_infosets_numba(
            infoset_node_indices,
            infoset_node_counts,
            node_reach,
            node_card_reach,
            node_hand_reach,
            infoset_opponent_reach,
            infoset_card_opponent_reach,
            infoset_hand_opponent_reach,
            infoset_node_hand_ratio,
            infoset_node_card_ratio,
            node_opponent_share,
            node_hand_opponent_reach,
        )
        return

    for infoset_id in range(infoset_count):
        _reduce_single_infoset(
            infoset_id=infoset_id,
            infoset_node_indices=infoset_node_indices,
            infoset_node_counts=infoset_node_counts,
            node_reach=node_reach,
            node_card_reach=node_card_reach,
            node_hand_reach=node_hand_reach,
            infoset_opponent_reach=infoset_opponent_reach,
            infoset_card_opponent_reach=infoset_card_opponent_reach,
            infoset_hand_opponent_reach=infoset_hand_opponent_reach,
            infoset_node_hand_ratio=infoset_node_hand_ratio,
            infoset_node_card_ratio=infoset_node_card_ratio,
            node_opponent_share=node_opponent_share,
            node_hand_opponent_reach=node_hand_opponent_reach,
        )


def _reduce_single_infoset(
    *,
    infoset_id: int,
    infoset_node_indices: np.ndarray,
    infoset_node_counts: np.ndarray,
    node_reach: np.ndarray,
    node_card_reach: np.ndarray,
    node_hand_reach: np.ndarray,
    infoset_opponent_reach: np.ndarray,
    infoset_card_opponent_reach: np.ndarray,
    infoset_hand_opponent_reach: np.ndarray,
    infoset_node_hand_ratio: np.ndarray,
    infoset_node_card_ratio: np.ndarray,
    node_opponent_share: np.ndarray,
    node_hand_opponent_reach: np.ndarray,
) -> None:
    count = int(infoset_node_counts[infoset_id])
    if count <= 0:
        raise ValueError("infoset must have at least one node")
    node_indices = infoset_node_indices[infoset_id, :count]
    if np.any(node_indices < 0):
        raise ValueError("infoset node indices must be dense")

    info_reach = node_reach[node_indices]
    info_card_reach = node_card_reach[node_indices]
    info_hand_reach = node_hand_reach[node_indices]

    infoset_opponent_reach[infoset_id] = np.sum(info_reach, dtype=np.float64)
    infoset_card_opponent_reach[infoset_id] = np.sum(info_card_reach, axis=0, dtype=np.float64)
    infoset_hand_opponent_reach[infoset_id] = np.sum(info_hand_reach, axis=0, dtype=np.float64)

    card_total = infoset_card_opponent_reach[infoset_id]
    hand_total = infoset_hand_opponent_reach[infoset_id]
    card_denominator = np.where(card_total > 0.0, card_total, 1.0)
    hand_denominator = np.where(hand_total > 0.0, hand_total, 1.0)

    infoset_node_card_ratio[infoset_id, :count] = np.divide(
        info_card_reach,
        card_denominator[None, :],
        out=np.zeros_like(info_card_reach),
        where=card_total[None, :] > 0.0,
    )
    infoset_node_hand_ratio[infoset_id, :count] = np.divide(
        info_hand_reach,
        hand_denominator[None, :],
        out=np.zeros_like(info_hand_reach),
        where=hand_total[None, :] > 0.0,
    )
    node_hand_opponent_reach[node_indices] = np.divide(
        info_hand_reach,
        hand_denominator[None, :],
        out=np.zeros_like(info_hand_reach),
        where=hand_total[None, :] > 0.0,
    )

    reach_total = float(infoset_opponent_reach[infoset_id])
    if reach_total > 0.0:
        node_opponent_share[node_indices] = info_reach / reach_total
    else:
        node_opponent_share[node_indices] = 1.0 / count


def _can_use_numba_parallel(max_workers: int | None) -> bool:
    return (
        njit is not None
        and prange is not None
        and (max_workers is None or max_workers > 1)
    )


if njit is not None and prange is not None:

    @njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _reduce_stage3_infosets_numba(
        infoset_node_indices: np.ndarray,
        infoset_node_counts: np.ndarray,
        node_reach: np.ndarray,
        node_card_reach: np.ndarray,
        node_hand_reach: np.ndarray,
        infoset_opponent_reach: np.ndarray,
        infoset_card_opponent_reach: np.ndarray,
        infoset_hand_opponent_reach: np.ndarray,
        infoset_node_hand_ratio: np.ndarray,
        infoset_node_card_ratio: np.ndarray,
        node_opponent_share: np.ndarray,
        node_hand_opponent_reach: np.ndarray,
    ) -> None:
        infoset_count = infoset_node_counts.shape[0]
        for infoset_id in prange(infoset_count):
            count = int(infoset_node_counts[infoset_id])
            if count <= 0:
                continue

            reach_total = 0.0
            for node_slot in range(count):
                node_index = int(infoset_node_indices[infoset_id, node_slot])
                reach_total += node_reach[node_index]

            infoset_opponent_reach[infoset_id] = reach_total

            for card_index in range(52):
                total = 0.0
                for node_slot in range(count):
                    node_index = int(infoset_node_indices[infoset_id, node_slot])
                    total += node_card_reach[node_index, card_index]
                infoset_card_opponent_reach[infoset_id, card_index] = total

            for hand_index in range(node_hand_reach.shape[1]):
                total = 0.0
                for node_slot in range(count):
                    node_index = int(infoset_node_indices[infoset_id, node_slot])
                    total += node_hand_reach[node_index, hand_index]
                infoset_hand_opponent_reach[infoset_id, hand_index] = total

            for node_slot in range(count):
                node_index = int(infoset_node_indices[infoset_id, node_slot])
                reach_value = node_reach[node_index]
                if reach_total > 0.0:
                    node_opponent_share[node_index] = reach_value / reach_total
                else:
                    node_opponent_share[node_index] = 1.0 / count

            for node_slot in range(count):
                node_index = int(infoset_node_indices[infoset_id, node_slot])
                for card_index in range(52):
                    total = infoset_card_opponent_reach[infoset_id, card_index]
                    if total > 0.0:
                        infoset_node_card_ratio[infoset_id, node_slot, card_index] = (
                            node_card_reach[node_index, card_index] / total
                        )
                    else:
                        infoset_node_card_ratio[infoset_id, node_slot, card_index] = 0.0
                for hand_index in range(node_hand_reach.shape[1]):
                    total = infoset_hand_opponent_reach[infoset_id, hand_index]
                    if total > 0.0:
                        ratio = node_hand_reach[node_index, hand_index] / total
                    else:
                        ratio = 0.0
                    infoset_node_hand_ratio[infoset_id, node_slot, hand_index] = ratio
                    node_hand_opponent_reach[node_index, hand_index] = ratio

else:

    def _reduce_stage3_infosets_numba(
        infoset_node_indices: np.ndarray,
        infoset_node_counts: np.ndarray,
        node_reach: np.ndarray,
        node_card_reach: np.ndarray,
        node_hand_reach: np.ndarray,
        infoset_opponent_reach: np.ndarray,
        infoset_card_opponent_reach: np.ndarray,
        infoset_hand_opponent_reach: np.ndarray,
        infoset_node_hand_ratio: np.ndarray,
        infoset_node_card_ratio: np.ndarray,
        node_opponent_share: np.ndarray,
        node_hand_opponent_reach: np.ndarray,
    ) -> None:
        raise RuntimeError("numba is not available")


