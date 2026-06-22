from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.spec import SolverStageResult
from pokergpu.core.board import Board


def test_run_solver_stage_runs_forward_prefix_branch_split_and_join(monkeypatch: pytest.MonkeyPatch) -> None:
    import pokergpu.cfr.solver.service as service

    tree = make_kuhn_public_tree()
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        cpu_workers=4,
        cpu_workers_stage3=2,
        cpu_workers_stage4=3,
        cpu_workers_stage6=5,
        cpu_workers_stage7=6,
    )
    dense_state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())
    calls: list[str] = []

    monkeypatch.setattr(
        service,
        "propagate_forward",
        lambda *args, **kwargs: _record_forward(calls, tree.node_count),
    )
    monkeypatch.setattr(
        service,
        "aggregate_prob_sum",
        lambda *args, **kwargs: _record_aggregate(calls, tree.node_count),
    )
    monkeypatch.setattr(
        service,
        "compute_opponent_reach",
        lambda *args, **kwargs: _record_opponent_reach(calls, tree.node_count, kwargs.get("max_workers")),
    )
    monkeypatch.setattr(
        service,
        "compute_showdown_equity",
        lambda *args, **kwargs: _record_showdown(calls, tree.node_count, kwargs.get("max_workers")),
    )
    monkeypatch.setattr(
        service,
        "evaluate_leaf_node_values",
        lambda *args, **kwargs: _record_leaf_values(calls, tree.node_count),
    )
    monkeypatch.setattr(
        service,
        "backward_cfv",
        lambda *args, **kwargs: _record_backward(calls, kwargs.get("max_workers")),
    )
    monkeypatch.setattr(
        service,
        "apply_dense_backward_cfv_update",
        lambda *args, **kwargs: _record_stage7(calls, dense_state, kwargs.get("max_workers")),
    )

    result = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
    )

    assert isinstance(result, SolverStageResult)
    assert result.request == request
    assert result.final_state == dense_state
    assert calls.count("stage1") == 1
    assert calls.count("stage2") == 1
    assert calls.count("stage3") == 1
    assert calls.count("stage4") == 0
    assert calls.count("stage5") == 1
    assert calls.count("stage6") == 1
    assert calls.count("stage7") == 1
    assert "cpu3:2" in calls
    assert "cpu6:5" in calls
    assert "cpu7:6" in calls


def test_run_solver_stage_reports_root_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    import pokergpu.cfr.solver.service as service

    tree = make_kuhn_public_tree()
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
    )
    dense_state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())

    monkeypatch.setattr(
        service,
        "propagate_forward",
        lambda *args, **kwargs: _record_forward([], tree.node_count),
    )
    monkeypatch.setattr(
        service,
        "aggregate_prob_sum",
        lambda *args, **kwargs: _record_aggregate([], tree.node_count),
    )
    monkeypatch.setattr(
        service,
        "compute_opponent_reach",
        lambda *args, **kwargs: _record_opponent_reach([], tree.node_count, kwargs.get("max_workers")),
    )
    monkeypatch.setattr(
        service,
        "compute_showdown_equity",
        lambda *args, **kwargs: _record_showdown([], tree.node_count, kwargs.get("max_workers")),
    )
    monkeypatch.setattr(
        service,
        "evaluate_leaf_node_values",
        lambda *args, **kwargs: _record_leaf_values([], tree.node_count),
    )
    monkeypatch.setattr(
        service,
        "backward_cfv",
        lambda *args, **kwargs: SimpleNamespace(
            node_values=tuple(0.0 for _ in range(tree.node_count)),
            infoset_values=np.asarray([1.25] + [0.0] * 5, dtype=np.float64),
            action_values=((1.0, -1.0),) + tuple(() for _ in range(tree.node_count - 1)),
        ),
    )
    monkeypatch.setattr(
        service,
        "apply_dense_backward_cfv_update",
        lambda *args, **kwargs: DenseCfrState(
            regret_sums=((0.5, -0.5),) + tuple((0.0, 0.0) for _ in range(5)),
            strategy_sums=((2.0, 1.0),) + tuple((0.0, 0.0) for _ in range(5)),
        ),
    )

    result = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
    )

    assert result.diagnostics is not None
    assert result.diagnostics["root_infoset"] == 0
    assert result.diagnostics["root_regrets"] == (0.5, -0.5)
    assert result.diagnostics["root_strategy_sums"] == (2.0, 1.0)
    assert result.diagnostics["root_action_values"] == (1.0, -1.0)
    assert result.diagnostics["root_node_value"] == 1.25


@pytest.mark.parametrize(
    "variant",
    [
        CfrVariant.CFR,
        CfrVariant.CFR_PLUS,
        CfrVariant.DCFR,
        CfrVariant.PREDICTIVE_CFR_PLUS,
    ],
)
def test_run_solver_stage_routes_cfr_variants_through_stage7(
    monkeypatch: pytest.MonkeyPatch,
    variant: CfrVariant,
) -> None:
    import pokergpu.cfr.solver.service as service

    tree = make_kuhn_public_tree()
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=variant,
        depth_limit=2,
    )
    dense_state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(6)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(6)),
    )
    board = Board(cards=())
    routed: list[CfrVariant] = []

    monkeypatch.setattr(service, "propagate_forward", lambda *args, **kwargs: _record_forward([], tree.node_count))
    monkeypatch.setattr(service, "aggregate_prob_sum", lambda *args, **kwargs: _record_aggregate([], tree.node_count))
    monkeypatch.setattr(
        service,
        "compute_opponent_reach",
        lambda *args, **kwargs: _record_opponent_reach([], tree.node_count, kwargs.get("max_workers")),
    )
    monkeypatch.setattr(
        service,
        "compute_showdown_equity",
        lambda *args, **kwargs: SimpleNamespace(
            node_showdown_equity=(0.0,) * tree.node_count,
            node_showdown_equity_bb=(0.0,) * tree.node_count,
            input_rows=SimpleNamespace(rows=()),
            output_rows=(),
        ),
    )
    monkeypatch.setattr(
        service,
        "evaluate_leaf_node_values",
        lambda *args, **kwargs: SimpleNamespace(node_values=(1.0,) * tree.node_count),
    )
    monkeypatch.setattr(service, "backward_cfv", lambda *args, **kwargs: SimpleNamespace())

    def fake_stage7(*args: object, **kwargs: object) -> DenseCfrState:
        routed.append(variant)
        return dense_state

    monkeypatch.setattr(service, "apply_dense_backward_cfv_update", fake_stage7)

    result = run_solver_stage(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
    )

    assert result.final_state == dense_state
    assert routed == [variant]


def test_seed_bias_perturbs_root_regrets(monkeypatch: pytest.MonkeyPatch) -> None:
    import pokergpu.cfr.solver.service as service
    from pokergpu.cfr.solver.infosets import build_dense_infoset_table

    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    dense_state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=2,
        seed=17,
    )

    biased = service._apply_seed_bias(request, dense_state, table)

    assert biased is not None
    assert biased.regret_sums[table.infoset_order[0]] != (0.0, 0.0)
    assert biased.strategy_sums[table.infoset_order[0]] != (0.0, 0.0)


def _record_call(calls: list[str], stage: str) -> SimpleNamespace:
    calls.append(stage)
    return SimpleNamespace()


def _record_forward(calls: list[str], node_count: int) -> SimpleNamespace:
    calls.append("stage1")
    return SimpleNamespace(
        node_reach=tuple(1.0 for _ in range(node_count)),
        infoset_reach=tuple(1.0 for _ in range(6)),
        action_reach=tuple(() for _ in range(node_count)),
    )


def _record_aggregate(calls: list[str], node_count: int) -> SimpleNamespace:
    calls.append("stage2")
    return SimpleNamespace(
        node_aggregate=SimpleNamespace(reach=tuple(1.0 for _ in range(node_count))),
        leaf_node_ids=tuple(),
    )


def _record_opponent_reach(calls: list[str], node_count: int, max_workers: int | None) -> SimpleNamespace:
    calls.append("stage3")
    calls.append(f"cpu3:{max_workers}")
    return SimpleNamespace(
        node_opponent_reach=tuple(1.0 for _ in range(node_count)),
        node_opponent_share=tuple(1.0 for _ in range(node_count)),
        node_hand_opponent_reach=tuple(() for _ in range(node_count)),
    )


def _record_showdown(calls: list[str], node_count: int, max_workers: int | None) -> SimpleNamespace:
    calls.append("stage4")
    calls.append(f"cpu4:{max_workers}")
    return SimpleNamespace(
        node_showdown_equity=(0.0,) * node_count,
        node_showdown_equity_bb=(0.0,) * node_count,
        input_rows=SimpleNamespace(rows=()),
        output_rows=(),
    )


def _record_leaf_values(calls: list[str], node_count: int) -> SimpleNamespace:
    calls.append("stage5")
    return SimpleNamespace(node_values=(1.0,) * node_count)


def _record_backward(calls: list[str], max_workers: int | None) -> SimpleNamespace:
    calls.append("stage6")
    calls.append(f"cpu6:{max_workers}")
    return SimpleNamespace()


def _record_stage7(calls: list[str], dense_state: DenseCfrState, max_workers: int | None) -> DenseCfrState:
    calls.append("stage7")
    calls.append(f"cpu7:{max_workers}")
    return dense_state
