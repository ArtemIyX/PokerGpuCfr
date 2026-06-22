from __future__ import annotations

from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from pokergpu.cfr.infosets import DenseInfosetTable
from pokergpu.cfr.stage6 import BackwardCFVResult
from .solver.state import DenseCfrState

__all__ = [
    "DenseCfrState",
    "apply_dense_backward_cfv_update",
    "regret_matching",
    "update_average_strategy",
    "update_regret",
]


def regret_matching(regrets: tuple[float, ...]) -> tuple[float, ...]:
    if not regrets:
        raise ValueError("regret vector cannot be empty")

    positive = tuple(max(0.0, regret) for regret in regrets)
    total = sum(positive)
    if total <= 0.0:
        return tuple(1.0 / len(regrets) for _ in regrets)
    return tuple(value / total for value in positive)


def update_regret(
    regret_sums: tuple[float, ...],
    action_values: tuple[float, ...],
    node_value: float,
) -> tuple[float, ...]:
    if len(regret_sums) != len(action_values):
        raise ValueError("regrets and action values must have the same length")
    return tuple(
        old_regret + (action_value - node_value)
        for old_regret, action_value in zip(regret_sums, action_values, strict=True)
    )


def update_average_strategy(
    strategy_sums: tuple[float, ...],
    strategy: tuple[float, ...],
    reach_weight: float,
) -> tuple[float, ...]:
    if len(strategy_sums) != len(strategy):
        raise ValueError("strategy sums and strategy must have the same length")
    return tuple(
        old_sum + reach_weight * action_prob
        for old_sum, action_prob in zip(strategy_sums, strategy, strict=True)
    )


def apply_dense_backward_cfv_update(
    state: DenseCfrState,
    backward: BackwardCFVResult,
    *,
    infoset_table: DenseInfosetTable,
    reach_weights: tuple[float, ...] | None = None,
    max_workers: int | None = None,
    executor: Executor | None = None,
) -> DenseCfrState:
    layout = _get_stage7_layout(infoset_table)
    if len(state.regret_sums) != infoset_table.infoset_count:
        raise ValueError("dense state must match infoset count")
    if len(backward.infoset_values) != infoset_table.infoset_count:
        raise ValueError("backward CFV infoset values must match infoset count")

    reach_vector = reach_weights or tuple(1.0 for _ in range(infoset_table.infoset_count))
    if len(reach_vector) != infoset_table.infoset_count:
        raise ValueError("reach weights must match infoset count")

    new_regrets: list[tuple[float, ...]] = [() for _ in range(infoset_table.infoset_count)]
    new_strategy_sums: list[tuple[float, ...]] = [() for _ in range(infoset_table.infoset_count)]

    infoset_ids = layout.infoset_ids
    if max_workers is None or max_workers <= 1 or len(infoset_ids) <= 1:
        for infoset_id in infoset_ids:
            _update_single_infoset(
                infoset_id=infoset_id,
                state=state,
                backward=backward,
                reach_vector=reach_vector,
                infoset_table=infoset_table,
                new_regrets=new_regrets,
                new_strategy_sums=new_strategy_sums,
            )
    elif executor is not None:
        chunks = _chunk_bounds(len(infoset_ids), max_workers)
        list(
            executor.map(
                lambda span: _update_infoset_chunk(
                    infoset_ids=infoset_ids[span[0] : span[1]],
                    state=state,
                    backward=backward,
                    reach_vector=reach_vector,
                    infoset_table=infoset_table,
                    new_regrets=new_regrets,
                    new_strategy_sums=new_strategy_sums,
                ),
                chunks,
            )
        )
    else:
        chunks = _chunk_bounds(len(infoset_ids), max_workers)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(infoset_ids))) as executor:
            list(
                executor.map(
                    lambda span: _update_infoset_chunk(
                        infoset_ids=infoset_ids[span[0] : span[1]],
                        state=state,
                        backward=backward,
                        reach_vector=reach_vector,
                        infoset_table=infoset_table,
                        new_regrets=new_regrets,
                        new_strategy_sums=new_strategy_sums,
                    ),
                    chunks,
                )
            )

    return DenseCfrState(
        regret_sums=tuple(new_regrets),
        strategy_sums=tuple(new_strategy_sums),
    )


@dataclass(slots=True, frozen=True)
class Stage7Layout:
    infoset_ids: tuple[int, ...]
    action_counts: tuple[int, ...]


@lru_cache(maxsize=32)
def _get_stage7_layout(infoset_table: DenseInfosetTable) -> Stage7Layout:
    return Stage7Layout(
        infoset_ids=tuple(
            infoset_id
            for infoset_id, node_index in enumerate(infoset_table.infoset_to_node)
            if node_index >= 0
        ),
        action_counts=infoset_table.action_counts,
    )


def _update_infoset_chunk(
    *,
    infoset_ids: tuple[int, ...],
    state: DenseCfrState,
    backward: BackwardCFVResult,
    reach_vector: tuple[float, ...],
    infoset_table: DenseInfosetTable,
    new_regrets: list[tuple[float, ...]],
    new_strategy_sums: list[tuple[float, ...]],
) -> None:
    for infoset_id in infoset_ids:
        _update_single_infoset(
            infoset_id=infoset_id,
            state=state,
            backward=backward,
            reach_vector=reach_vector,
            infoset_table=infoset_table,
            new_regrets=new_regrets,
            new_strategy_sums=new_strategy_sums,
        )


def _update_single_infoset(
    *,
    infoset_id: int,
    state: DenseCfrState,
    backward: BackwardCFVResult,
    reach_vector: tuple[float, ...],
    infoset_table: DenseInfosetTable,
    new_regrets: list[tuple[float, ...]],
    new_strategy_sums: list[tuple[float, ...]],
) -> None:
    node_index = infoset_table.infoset_to_node[infoset_id]
    if node_index < 0:
        return
    action_count = infoset_table.action_counts[infoset_id]
    regrets = state.regret_sums[infoset_id]
    strategy_sums = state.strategy_sums[infoset_id]
    values = backward.action_values[node_index]
    if len(values) != action_count:
        raise AssertionError(
            "stage7 width mismatch: "
            f"infoset_id={infoset_id} node_index={node_index} "
            f"action_count={action_count} action_values={len(values)}"
        )
    if len(regrets) != action_count:
        raise ValueError("action values must match regret row width")
    if len(strategy_sums) != action_count:
        raise ValueError("strategy row must match action value width")

    node_value = float(backward.infoset_values[infoset_id])
    reach_weight = float(reach_vector[infoset_id])

    positive_total = 0.0
    for regret in regrets:
        if regret > 0.0:
            positive_total += regret

    updated_regrets = [0.0] * action_count
    updated_strategy_sums = [0.0] * action_count
    if positive_total <= 0.0:
        uniform_prob = 1.0 / action_count
        for action_index in range(action_count):
            updated_regrets[action_index] = (
                regrets[action_index] + (float(values[action_index]) - node_value)
            )
            updated_strategy_sums[action_index] = (
                strategy_sums[action_index] + reach_weight * uniform_prob
            )
    else:
        for action_index in range(action_count):
            old_regret = regrets[action_index]
            strategy = old_regret / positive_total if old_regret > 0.0 else 0.0
            updated_regrets[action_index] = (
                old_regret + (float(values[action_index]) - node_value)
            )
            updated_strategy_sums[action_index] = (
                strategy_sums[action_index] + reach_weight * strategy
            )

    new_regrets[infoset_id] = tuple(updated_regrets)
    new_strategy_sums[infoset_id] = tuple(updated_strategy_sums)


def _chunk_bounds(count: int, workers: int) -> tuple[tuple[int, int], ...]:
    if count <= 0:
        return ()
    chunk_size = (count + workers - 1) // workers
    return tuple((start, min(start + chunk_size, count)) for start in range(0, count, chunk_size))
