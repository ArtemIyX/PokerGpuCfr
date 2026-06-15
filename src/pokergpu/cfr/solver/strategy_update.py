from __future__ import annotations

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
) -> DenseCfrState:
    assert infoset_table.infoset_count == len(state.regret_sums), "infoset table must align with state"
    if len(state.strategy_sums) != len(state.regret_sums):
        raise ValueError("dense strategy and regret tables must have the same size")
    if len(action_values) != len(state.regret_sums):
        raise ValueError("action values must match infoset count")

    reach_vector = reach_weights or tuple(1.0 for _ in state.regret_sums)
    if len(reach_vector) != len(state.regret_sums):
        raise ValueError("reach weights must match infoset count")

    new_regrets: list[tuple[float, ...]] = []
    new_strategy_sums: list[tuple[float, ...]] = []

    for infoset_id in infoset_table.infoset_order:
        regrets = state.regret_sums[infoset_id]
        values = action_values[infoset_id]
        strategy_sums = state.strategy_sums[infoset_id]
        if len(regrets) != len(values):
            raise ValueError("action values must match regret row width")
        if len(strategy_sums) != len(values):
            raise ValueError("strategy row must match action value width")

        strategy = normalize_strategy(regret_matching(regrets))
        node_value = sum(prob * value for prob, value in zip(strategy, values, strict=True))
        new_regrets.append(update_regret(regrets, values, node_value))
        new_strategy_sums.append(
            update_average_strategy(strategy_sums, strategy, reach_vector[infoset_id])
        )

    return DenseCfrState(
        regret_sums=tuple(new_regrets),
        strategy_sums=tuple(new_strategy_sums),
    )
