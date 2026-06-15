from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import ceil

from .infosets import DenseInfosetTable
from .state import DenseCfrState, SolverIterationResult, SolverState
from ..stage1 import normalize_strategy
from ..stage7 import regret_matching, update_average_strategy, update_regret


def apply_solver_strategy_update(
    state: SolverState,
    action_values: tuple[float, ...],
    *,
    reach_weight: float = 1.0,
) -> SolverIterationResult:
    assert reach_weight >= 0.0, "reach weight must be non-negative"
    if not action_values:
        raise ValueError("action values cannot be empty")
    if len(state.regret_sums) != len(action_values):
        raise ValueError("state and action values must have the same length")
    if len(state.strategy_sums) != len(action_values):
        raise ValueError("state and action values must have the same length")

    strategy = normalize_strategy(regret_matching(state.regret_sums))
    assert len(strategy) == len(action_values), "strategy and action values must align"
    node_value = sum(prob * value for prob, value in zip(strategy, action_values, strict=True))
    regret_sums = update_regret(state.regret_sums, action_values, node_value)
    strategy_sums = update_average_strategy(state.strategy_sums, strategy, reach_weight)

    return SolverIterationResult(
        state=SolverState(regret_sums=regret_sums, strategy_sums=strategy_sums),
        strategy=strategy,
        node_value=node_value,
        action_values=action_values,
    )


def apply_dense_solver_strategy_update(
    state: DenseCfrState,
    action_values: tuple[tuple[float, ...], ...],
    *,
    infoset_table: DenseInfosetTable,
    reach_weights: tuple[float, ...] | None = None,
    max_workers: int | None = None,
) -> DenseCfrState:
    assert infoset_table.infoset_count == len(state.regret_sums), "infoset table must align with state"
    if len(state.strategy_sums) != len(state.regret_sums):
        raise ValueError("dense strategy and regret tables must have the same size")
    if len(action_values) != len(state.regret_sums):
        raise ValueError("action values must match infoset count")

    reach_vector = reach_weights or tuple(1.0 for _ in state.regret_sums)
    if len(reach_vector) != len(state.regret_sums):
        raise ValueError("reach weights must match infoset count")

    def process_infoset(infoset_id: int) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
        regrets = state.regret_sums[infoset_id]
        values = action_values[infoset_id]
        strategy_sums = state.strategy_sums[infoset_id]
        if len(regrets) != len(values):
            raise ValueError("action values must match regret row width")
        if len(strategy_sums) != len(values):
            raise ValueError("strategy row must match action value width")

        strategy = normalize_strategy(regret_matching(regrets))
        node_value = sum(prob * value for prob, value in zip(strategy, values, strict=True))
        return (
            infoset_id,
            update_regret(regrets, values, node_value),
            update_average_strategy(strategy_sums, strategy, reach_vector[infoset_id]),
        )

    infoset_ids = list(infoset_table.infoset_order)
    if max_workers is None or max_workers <= 1 or len(infoset_ids) <= 1:
        results = [process_infoset(infoset_id) for infoset_id in infoset_ids]
    else:
        chunk_size = max(1, ceil(len(infoset_ids) / max_workers))
        chunks = [tuple(infoset_ids[index : index + chunk_size]) for index in range(0, len(infoset_ids), chunk_size)]
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chunk_result in executor.map(lambda chunk: tuple(process_infoset(infoset_id) for infoset_id in chunk), chunks):
                results.extend(chunk_result)

    new_regrets: list[tuple[float, ...]] = [() for _ in range(len(state.regret_sums))]
    new_strategy_sums: list[tuple[float, ...]] = [() for _ in range(len(state.strategy_sums))]
    for infoset_id, regrets, strategy_sums in results:
        new_regrets[infoset_id] = regrets
        new_strategy_sums[infoset_id] = strategy_sums

    return DenseCfrState(
        regret_sums=tuple(new_regrets),
        strategy_sums=tuple(new_strategy_sums),
    )
