from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.infosets import DenseInfosetTable
from pokergpu.cfr.infosets import build_dense_infoset_table
from pokergpu.abstraction.hands import private_hand_count
from pokergpu.tree.public_tree import PublicTree

try:
    from numba import njit, prange  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    njit = None
    prange = None


STAGE3_INFOSET_BLOCK = 8


@dataclass(slots=True, frozen=True)
class OpponentReachResult:
    """Blocking-aware opponent reach summaries for a single public tree."""

    infoset_opponent_reach: np.ndarray
    infoset_card_opponent_reach: np.ndarray
    infoset_hand_opponent_reach: np.ndarray
    infoset_node_hand_ratio: np.ndarray
    infoset_node_card_ratio: np.ndarray
    node_opponent_reach: np.ndarray
    node_opponent_share: np.ndarray
    node_hand_opponent_reach: np.ndarray

    def to_legacy(self, infoset_table: DenseInfosetTable) -> tuple[
        tuple[float, ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[tuple[float, ...], ...], ...],
        tuple[tuple[tuple[float, ...], ...], ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[tuple[float, ...], ...],
    ]:
        infoset_counts = infoset_table.infoset_node_counts
        row_offsets = _build_row_offsets(infoset_counts)
        return (
            tuple(float(value) for value in self.infoset_opponent_reach),
            tuple(tuple(float(value) for value in row) for row in self.infoset_card_opponent_reach),
            tuple(tuple(float(value) for value in row) for row in self.infoset_hand_opponent_reach),
            tuple(
                tuple(
                    tuple(float(value) for value in node_row)
                    for node_row in _iter_infoset_rows(
                        self.infoset_node_hand_ratio,
                        row_offsets,
                        infoset_id,
                        int(infoset_counts[infoset_id]),
                    )
                )
                for infoset_id in range(infoset_table.infoset_count)
            ),
            tuple(
                tuple(
                    tuple(float(value) for value in node_row)
                    for node_row in _iter_infoset_rows(
                        self.infoset_node_card_ratio,
                        row_offsets,
                        infoset_id,
                        int(infoset_counts[infoset_id]),
                    )
                )
                for infoset_id in range(infoset_table.infoset_count)
            ),
            tuple(float(value) for value in self.node_opponent_reach),
            tuple(float(value) for value in self.node_opponent_share),
            tuple(tuple(float(value) for value in row) for row in self.node_hand_opponent_reach),
        )


@dataclass(slots=True, frozen=True)
class Stage3PreparedInput:
    infoset_table: DenseInfosetTable
    infoset_node_indices: np.ndarray
    infoset_node_counts: np.ndarray
    infoset_row_offsets: np.ndarray
    block_infoset_starts: np.ndarray
    block_infoset_ends: np.ndarray
    node_reach: np.ndarray
    node_card_reach: np.ndarray
    node_hand_reach: np.ndarray


@dataclass(slots=True, frozen=True)
class Stage3Layout:
    infoset_table: DenseInfosetTable
    infoset_node_indices: np.ndarray
    infoset_node_counts: np.ndarray
    infoset_row_offsets: np.ndarray
    block_infoset_starts: np.ndarray
    block_infoset_ends: np.ndarray


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
    total_infoset_nodes = int(prepared.infoset_row_offsets[-1]) if infoset_count > 0 else 0
    infoset_node_hand_ratio = np.zeros((total_infoset_nodes, hand_width), dtype=np.float64)
    infoset_node_card_ratio = np.zeros((total_infoset_nodes, 52), dtype=np.float64)

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
        infoset_row_offsets=prepared.infoset_row_offsets,
        block_infoset_starts=prepared.block_infoset_starts,
        block_infoset_ends=prepared.block_infoset_ends,
        node_opponent_share=node_opponent_share,
        node_hand_opponent_reach=node_hand_opponent_reach,
        max_workers=max_workers,
    )

    return OpponentReachResult(
        infoset_opponent_reach=infoset_opponent_reach,
        infoset_card_opponent_reach=infoset_card_opponent_reach,
        infoset_hand_opponent_reach=infoset_hand_opponent_reach,
        infoset_node_hand_ratio=infoset_node_hand_ratio,
        infoset_node_card_ratio=infoset_node_card_ratio,
        node_opponent_reach=node_reach,
        node_opponent_share=node_opponent_share,
        node_hand_opponent_reach=node_hand_opponent_reach,
    )


def _prepare_stage3_input(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
) -> Stage3PreparedInput:
    """Cache-backed dense infoset layout and rectangular Stage 3 buffers."""
    layout = _get_stage3_layout(tree)
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

    return Stage3PreparedInput(
        infoset_table=layout.infoset_table,
        infoset_node_indices=layout.infoset_node_indices,
        infoset_node_counts=layout.infoset_node_counts,
        infoset_row_offsets=layout.infoset_row_offsets,
        block_infoset_starts=layout.block_infoset_starts,
        block_infoset_ends=layout.block_infoset_ends,
        node_reach=node_reach,
        node_card_reach=node_card_reach,
        node_hand_reach=node_hand_reach,
    )


@lru_cache(maxsize=32)
def _get_stage3_layout(tree: PublicTree) -> Stage3Layout:
    infoset_table = build_dense_infoset_table(tree)
    infoset_count = infoset_table.infoset_count
    infoset_node_counts = np.asarray(infoset_table.infoset_node_counts, dtype=np.int64)
    infoset_row_offsets = _build_row_offsets(infoset_table.infoset_node_counts)
    total_infoset_nodes = int(infoset_row_offsets[-1]) if infoset_count > 0 else 0
    infoset_node_indices = np.empty(total_infoset_nodes, dtype=np.int64)
    running_offset = 0
    for nodes in infoset_table.infoset_nodes:
        count = len(nodes)
        if count > 0:
            infoset_node_indices[running_offset : running_offset + count] = np.asarray(nodes, dtype=np.int64)
        running_offset += count
    block_infoset_starts, block_infoset_ends = _build_stage3_blocks(infoset_node_counts)
    return Stage3Layout(
        infoset_table=infoset_table,
        infoset_node_indices=infoset_node_indices,
        infoset_node_counts=infoset_node_counts,
        infoset_row_offsets=infoset_row_offsets,
        block_infoset_starts=block_infoset_starts,
        block_infoset_ends=block_infoset_ends,
    )


def _reduce_stage3_infosets(
    *,
    infoset_node_indices: np.ndarray,
    infoset_node_counts: np.ndarray,
    infoset_row_offsets: np.ndarray,
    block_infoset_starts: np.ndarray,
    block_infoset_ends: np.ndarray,
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
            infoset_row_offsets,
            block_infoset_starts,
            block_infoset_ends,
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
            infoset_row_offsets=infoset_row_offsets,
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


def _build_stage3_blocks(infoset_node_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    infoset_count = int(infoset_node_counts.shape[0])
    if infoset_count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    total_nodes = int(np.sum(infoset_node_counts, dtype=np.int64))
    if total_nodes <= 0:
        empty_blocks = np.arange(infoset_count, dtype=np.int64)
        return empty_blocks, empty_blocks.copy()

    target_blocks = max(1, min(infoset_count, total_nodes // max(1, STAGE3_INFOSET_BLOCK * 4)))
    if target_blocks == 1:
        return np.array([0], dtype=np.int64), np.array([infoset_count], dtype=np.int64)

    target_work = max(1, (total_nodes + target_blocks - 1) // target_blocks)
    block_starts: list[int] = []
    block_ends: list[int] = []
    block_start = 0
    running_work = 0
    for infoset_id in range(infoset_count):
        count = int(infoset_node_counts[infoset_id])
        if infoset_id > block_start and running_work >= target_work:
            block_starts.append(block_start)
            block_ends.append(infoset_id)
            block_start = infoset_id
            running_work = 0
        running_work += count
    block_starts.append(block_start)
    block_ends.append(infoset_count)
    return np.asarray(block_starts, dtype=np.int64), np.asarray(block_ends, dtype=np.int64)


def _reduce_single_infoset(
    *,
    infoset_id: int,
    infoset_node_indices: np.ndarray,
    infoset_node_counts: np.ndarray,
    infoset_row_offsets: np.ndarray,
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
    row_offset = int(infoset_row_offsets[infoset_id])
    node_indices = infoset_node_indices[row_offset : row_offset + count]
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

    infoset_node_card_ratio[row_offset : row_offset + count] = np.divide(
        info_card_reach,
        card_denominator[None, :],
        out=np.zeros_like(info_card_reach),
        where=card_total[None, :] > 0.0,
    )
    infoset_node_hand_ratio[row_offset : row_offset + count] = np.divide(
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
        infoset_row_offsets: np.ndarray,
        block_infoset_starts: np.ndarray,
        block_infoset_ends: np.ndarray,
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
        block_count = block_infoset_starts.shape[0]
        for block_index in prange(block_count):
            block_start = int(block_infoset_starts[block_index])
            block_stop = int(block_infoset_ends[block_index])

            for infoset_id in range(block_start, block_stop):
                count = int(infoset_node_counts[infoset_id])
                if count <= 0:
                    continue

                row_offset = int(infoset_row_offsets[infoset_id])
                reach_total = 0.0
                for node_slot in range(count):
                    node_index = int(infoset_node_indices[row_offset + node_slot])
                    reach_total += node_reach[node_index]

                infoset_opponent_reach[infoset_id] = reach_total

                for card_index in range(52):
                    total = 0.0
                    for node_slot in range(count):
                        node_index = int(infoset_node_indices[row_offset + node_slot])
                        total += node_card_reach[node_index, card_index]
                    infoset_card_opponent_reach[infoset_id, card_index] = total

                for hand_index in range(node_hand_reach.shape[1]):
                    total = 0.0
                    for node_slot in range(count):
                        node_index = int(infoset_node_indices[row_offset + node_slot])
                        total += node_hand_reach[node_index, hand_index]
                    infoset_hand_opponent_reach[infoset_id, hand_index] = total

                for node_slot in range(count):
                    node_index = int(infoset_node_indices[row_offset + node_slot])
                    reach_value = node_reach[node_index]
                    if reach_total > 0.0:
                        node_opponent_share[node_index] = reach_value / reach_total
                    else:
                        node_opponent_share[node_index] = 1.0 / count

                for node_slot in range(count):
                    node_index = int(infoset_node_indices[row_offset + node_slot])
                    flat_row = row_offset + node_slot
                    for card_index in range(52):
                        total = infoset_card_opponent_reach[infoset_id, card_index]
                        if total > 0.0:
                            infoset_node_card_ratio[flat_row, card_index] = (
                                node_card_reach[node_index, card_index] / total
                            )
                        else:
                            infoset_node_card_ratio[flat_row, card_index] = 0.0
                    for hand_index in range(node_hand_reach.shape[1]):
                        total = infoset_hand_opponent_reach[infoset_id, hand_index]
                        if total > 0.0:
                            ratio = node_hand_reach[node_index, hand_index] / total
                        else:
                            ratio = 0.0
                        infoset_node_hand_ratio[flat_row, hand_index] = ratio
                        node_hand_opponent_reach[node_index, hand_index] = ratio

else:

    def _reduce_stage3_infosets_numba(
        infoset_node_indices: np.ndarray,
        infoset_node_counts: np.ndarray,
        infoset_row_offsets: np.ndarray,
        block_infoset_starts: np.ndarray,
        block_infoset_ends: np.ndarray,
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


def _iter_infoset_rows(
    array: np.ndarray,
    row_offsets: np.ndarray,
    infoset_id: int,
    count: int,
) -> tuple[np.ndarray, ...]:
    row_offset = int(row_offsets[infoset_id])
    return tuple(array[row_offset + node_index] for node_index in range(count))


def _build_row_offsets(infoset_counts: tuple[int, ...]) -> np.ndarray:
    row_offsets = np.zeros(len(infoset_counts) + 1, dtype=np.int64)
    running_offset = 0
    for infoset_id, count in enumerate(infoset_counts):
        row_offsets[infoset_id] = running_offset
        running_offset += count
    row_offsets[len(infoset_counts)] = running_offset
    return row_offsets


