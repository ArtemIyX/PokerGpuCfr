from __future__ import annotations

from pokergpu.cfr.infosets import DenseInfosetTable
from pokergpu.cfr.stage1 import normalize_strategy
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
) -> DenseCfrState:
    if len(state.regret_sums) != infoset_table.infoset_count:
        raise ValueError("dense state must match infoset count")
    if len(backward.infoset_values) != infoset_table.infoset_count:
        raise ValueError("backward CFV infoset values must match infoset count")

    reach_vector = reach_weights or tuple(1.0 for _ in range(infoset_table.infoset_count))
    if len(reach_vector) != infoset_table.infoset_count:
        raise ValueError("reach weights must match infoset count")

    new_regrets: list[tuple[float, ...]] = [() for _ in range(infoset_table.infoset_count)]
    new_strategy_sums: list[tuple[float, ...]] = [() for _ in range(infoset_table.infoset_count)]

    for infoset_id, node_index in enumerate(infoset_table.infoset_to_node):
        if node_index < 0:
            continue
        regrets = state.regret_sums[infoset_id]
        strategy_sums = state.strategy_sums[infoset_id]
        values = backward.action_values[node_index]
        if len(regrets) != len(values):
            raise ValueError("action values must match regret row width")
        if len(strategy_sums) != len(values):
            raise ValueError("strategy row must match action value width")

        node_value = float(backward.infoset_values[infoset_id])
        strategy = normalize_strategy(regret_matching(regrets))
        new_regrets[infoset_id] = update_regret(regrets, values, node_value)
        new_strategy_sums[infoset_id] = update_average_strategy(
            strategy_sums,
            strategy,
            reach_vector[infoset_id],
        )

    return DenseCfrState(
        regret_sums=tuple(new_regrets),
        strategy_sums=tuple(new_strategy_sums),
    )
