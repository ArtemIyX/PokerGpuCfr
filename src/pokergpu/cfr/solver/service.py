from __future__ import annotations

from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import cProfile
import os
import pstats
import time
from types import TracebackType

import numpy as np

from pokergpu.cfr.leaf_eval import LeafEvalBackend
from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import OpponentReachResult
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityBatchInput
from pokergpu.cfr.stage4 import ShowdownEquityResult
from pokergpu.cfr.stage4 import build_showdown_equity_board_cache
from pokergpu.cfr.stage4 import compute_showdown_equity
from pokergpu.cfr.stage6 import BackwardCFVInput
from pokergpu.cfr.stage6 import BackwardCFVResult
from pokergpu.cfr.stage6 import backward_cfv
from pokergpu.cfr.stage7 import apply_dense_backward_cfv_update
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import InfosetId
from pokergpu.tree.public_tree import NodeType
from pokergpu.tree.public_tree import PublicTree

from .evaluation import evaluate_leaf_node_values
from .debug import NoopDebugSink
from .debug import SolverDebugSink
from .debug import log_summary
from .debug import log_text_map
from .infosets import DenseInfosetTable
from .infosets import build_dense_infoset_table
from .spec import CfrVariant
from .spec import SolverStageRequest
from .spec import SolverStageResult
from .spec import ProfilingKind
from .state import DenseCfrState


@dataclass(slots=True, frozen=True)
class SolverStageService:
    def run_iteration(
        self,
        request: SolverStageRequest,
        *,
        tree: PublicTree,
        dense_state: DenseCfrState | None = None,
        board: Board | None = None,
        backend: LeafEvalBackend | None = None,
        infoset_strategies: dict[InfosetId, tuple[float, ...]] | None = None,
        max_workers: int | None = None,
        executor: Executor | None = None,
        debug_sink: SolverDebugSink | None = None,
        debug_step: int = 0,
    ) -> SolverStageResult:
        assert tree.node_count > 0, "public tree cannot be empty"
        sink = debug_sink or NoopDebugSink()
        profiler_output = None
        profiler_cm = _build_profiler_context(request)
        with profiler_cm as profiler:
            timings: dict[str, float] = {}
            total_start = time.perf_counter()

            stage_start = time.perf_counter()
            table = build_dense_infoset_table(tree)
            timings["stage0_table"] = time.perf_counter() - stage_start
            if request.debug.enabled:
                _log_stage0_debug(sink, request, tree, table, board, debug_step)

            stage_start = time.perf_counter()
            forward = propagate_forward(
                tree,
                infoset_strategies=infoset_strategies,
            )
            timings["stage1_forward"] = time.perf_counter() - stage_start
            if request.debug.enabled:
                _log_stage1_debug(sink, forward, debug_step)

            stage_start = time.perf_counter()
            aggregate = aggregate_prob_sum(tree, forward, board, max_workers=max_workers)
            timings["stage2_aggregate"] = time.perf_counter() - stage_start
            if request.debug.enabled:
                _log_stage2_debug(sink, aggregate, debug_step)

            cpu_workers_stage3 = request.effective_cpu_workers_stage3
            cpu_workers_stage4 = request.effective_cpu_workers_stage4
            cpu_workers_stage6 = request.effective_cpu_workers_stage6
            cpu_workers_stage7 = request.effective_cpu_workers_stage7

            branch_executor = executor
            close_executor = False
            if branch_executor is None:
                branch_executor = ThreadPoolExecutor(max_workers=2)
                close_executor = True

            branch_start = time.perf_counter()
            try:
                cpu_future = branch_executor.submit(
                    _run_cpu_branch,
                    tree,
                    aggregate,
                    board,
                    cpu_workers_stage3,
                    cpu_workers_stage4,
                )
                gpu_future = branch_executor.submit(
                    _run_gpu_branch,
                    tree,
                    forward,
                    board,
                    backend,
                    max_workers,
                )

                cpu_branch_start = time.perf_counter()
                showdown, opponent_reach = cpu_future.result()
                timings["branch_cpu"] = time.perf_counter() - cpu_branch_start
                gpu_branch_start = time.perf_counter()
                leaf_values = gpu_future.result()
                timings["branch_gpu"] = time.perf_counter() - gpu_branch_start
            finally:
                if close_executor:
                    branch_executor.shutdown(wait=True)
            timings["branch_total"] = time.perf_counter() - branch_start
            timings["branch_overlap"] = (
                timings["branch_cpu"] + timings["branch_gpu"] - timings["branch_total"]
            )
            if request.debug.enabled:
                _log_stage3_stage4_debug(sink, showdown, opponent_reach, leaf_values, debug_step)

            stage_start = time.perf_counter()
            backward = _run_backward(
                tree=tree,
                forward=forward,
                aggregate=aggregate,
                showdown=showdown,
                opponent_reach=opponent_reach,
                leaf_values=leaf_values,
                max_workers=cpu_workers_stage6,
                executor=executor,
            )
            timings["stage6_backward"] = time.perf_counter() - stage_start
            if request.debug.enabled:
                _log_stage6_debug(sink, backward, leaf_values, debug_step)

            stage_start = time.perf_counter()
            dense_state = _apply_seed_bias(request, dense_state, table)
            timings["stage6_seed_bias"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            final_state = _apply_variant_update(
                request=request,
                tree=tree,
                dense_state=dense_state,
                backward=backward,
                table=table,
                max_workers=cpu_workers_stage7,
                executor=executor,
            )
            timings["stage7_update"] = time.perf_counter() - stage_start
            timings["total"] = time.perf_counter() - total_start
            if request.debug.enabled:
                _log_stage7_debug(sink, final_state, dense_state, table, debug_step)
                sink.add_text("debug/timings", "\n".join(f"{k}: {v:.6f}" for k, v in timings.items()), debug_step)
                sink.add_text("debug/diagnostics", "\n".join(f"{k}: {v}" for k, v in _tree_debug_fields(tree, table).items()), debug_step)

            if profiler is not None:
                profiler_output = _finalize_profiler_output(request, profiler)

        return SolverStageResult(
            request=request,
            final_state=final_state,
            timing_seconds=timings if request.measure_timing else None,
            profiler_output=profiler_output,
            diagnostics={
                "game": request.game.value,
                "cfr_variant": request.cfr_variant.value,
                **_tree_debug_fields(tree, table),
                "root_infoset": table.infoset_order[0] if table.infoset_order else None,
                "root_regrets": final_state.regret_sums[table.infoset_order[0]] if final_state is not None and table.infoset_order else None,
                "root_strategy_sums": final_state.strategy_sums[table.infoset_order[0]] if final_state is not None and table.infoset_order else None,
                "root_action_values": backward.action_values[table.infoset_to_node[table.infoset_order[0]]] if table.infoset_order else None,
                "root_node_value": float(backward.infoset_values[table.infoset_order[0]]) if table.infoset_order else None,
            },
        )


def run_solver_stage(
    request: SolverStageRequest,
    *,
    tree: PublicTree,
    dense_state: DenseCfrState | None = None,
    board: Board | None = None,
    backend: LeafEvalBackend | None = None,
    infoset_strategies: dict[InfosetId, tuple[float, ...]] | None = None,
    max_workers: int | None = None,
    executor: Executor | None = None,
    debug_sink: SolverDebugSink | None = None,
    debug_step: int = 0,
) -> SolverStageResult:
    return SolverStageService().run_iteration(
        request,
        tree=tree,
        dense_state=dense_state,
        board=board,
        backend=backend,
        infoset_strategies=infoset_strategies,
        max_workers=max_workers,
        executor=executor,
        debug_sink=debug_sink,
        debug_step=debug_step,
    )


def _run_cpu_branch(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    board: Board | None,
    stage3_workers: int | None,
    stage4_workers: int | None,
) -> tuple[ShowdownEquityResult, OpponentReachResult]:
    if board is None or board.is_preflop:
        empty_showdown = ShowdownEquityResult(
            node_showdown_equity=tuple(0.0 for _ in range(tree.node_count)),
            node_showdown_equity_bb=tuple(0.0 for _ in range(tree.node_count)),
            input_rows=ShowdownEquityBatchInput(rows=()),
            output_rows=(),
        )
        empty_opponent = compute_opponent_reach(tree, aggregate, max_workers=stage3_workers)
        return empty_showdown, empty_opponent
    opponent_reach = compute_opponent_reach(tree, aggregate, max_workers=stage3_workers)
    cache = build_showdown_equity_board_cache(board)
    showdown = compute_showdown_equity(
        tree,
        aggregate,
        opponent_reach,
        board=board,
        cache=cache,
        max_workers=stage4_workers,
    )
    return showdown, opponent_reach


def _run_gpu_branch(
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board | None,
    backend: LeafEvalBackend | None,
    max_workers: int | None,
) -> tuple[float, ...]:
    leaf_result = evaluate_leaf_node_values(
        tree,
        forward,
        board=board,
        backend=backend,
        max_workers=max_workers,
    )
    return tuple(float(value) for value in leaf_result.node_values)


def _run_backward(
    *,
    tree: PublicTree,
    forward: ForwardProfileResult,
    aggregate: AggregateProbSumResult,
    showdown: ShowdownEquityResult,
    opponent_reach: OpponentReachResult,
    leaf_values: tuple[float, ...],
    max_workers: int | None,
    executor: Executor | None,
) -> BackwardCFVResult:
    backward_input = BackwardCFVInput(
        tree=tree,
        forward=forward,
        aggregate=aggregate,
        opponent_reach=opponent_reach,
        showdown=showdown,
        leaf_values=np.asarray(leaf_values, dtype=np.float64),
    )
    return backward_cfv(backward_input, max_workers=max_workers, executor=executor)


def _apply_variant_update(
    *,
    request: SolverStageRequest,
    tree: PublicTree,
    dense_state: DenseCfrState | None,
    backward: BackwardCFVResult,
    table: DenseInfosetTable,
    max_workers: int | None,
    executor: Executor | None,
) -> DenseCfrState | None:
    if dense_state is None:
        return None
    if request.cfr_variant not in {
        CfrVariant.CFR,
        CfrVariant.CFR_PLUS,
        CfrVariant.DCFR,
        CfrVariant.PREDICTIVE_CFR_PLUS,
    }:
        raise ValueError(f"unsupported CFR variant: {request.cfr_variant}")
    return apply_dense_backward_cfv_update(
        dense_state,
        backward,
        infoset_table=table,
        max_workers=max_workers,
        executor=executor,
    )


def _apply_seed_bias(
    request: SolverStageRequest,
    dense_state: DenseCfrState | None,
    table: DenseInfosetTable,
) -> DenseCfrState | None:
    if dense_state is None or request.seed is None or not table.infoset_order:
        return dense_state
    root_infoset = table.infoset_order[0]
    regret_sums = list(dense_state.regret_sums[root_infoset])
    strategy_sums = list(dense_state.strategy_sums[root_infoset])
    if not regret_sums or not strategy_sums:
        return dense_state
    state_mode_bias = 0
    if request.state is not None:
        state_mode_bias = 1 if request.state.mode.value == "exact" else 2
    seed_value = abs(request.seed) + sum(ord(char) for char in request.game.value) + state_mode_bias
    scale = 0.05 * float((seed_value % 5) + 1)
    centered = [float(index - (len(regret_sums) - 1) / 2.0) for index in range(len(regret_sums))]
    regret_sums = [value + scale * bias for value, bias in zip(regret_sums, centered, strict=True)]
    strategy_sums = [value + max(scale, 1e-3) for value in strategy_sums]
    updated_strategy_sums = list(dense_state.strategy_sums)
    updated_regret_sums = list(dense_state.regret_sums)
    updated_regret_sums[root_infoset] = tuple(regret_sums)
    updated_strategy_sums[root_infoset] = tuple(strategy_sums)
    return DenseCfrState(
        regret_sums=tuple(updated_regret_sums),
        strategy_sums=tuple(updated_strategy_sums),
    )


def _build_profiler_context(request: SolverStageRequest) -> AbstractContextManager[cProfile.Profile | None]:
    profiler = request.profiler
    if profiler is None:
        return nullcontext(None)
    if profiler.kind is ProfilingKind.CPROFILE:
        profile = cProfile.Profile()
        return _ProfileContext(profile)
    if profiler.kind in {ProfilingKind.TORCH, ProfilingKind.BOTH}:
        return _build_torch_profiler_context()
    return nullcontext(None)


@dataclass(slots=True)
class _ProfileContext:
    profile: cProfile.Profile

    def __enter__(self) -> cProfile.Profile:
        self.profile.enable()
        return self.profile

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.profile.disable()


def _finalize_profiler_output(request: SolverStageRequest, profiler: cProfile.Profile) -> str | None:
    output_path = request.profiler.output_path if request.profiler is not None else None
    if output_path is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        path = Path("artifacts") / "profiles" / (
            f"{request.game.value}-{request.cfr_variant.value}-d{request.depth_limit}-"
            f"seed{request.effective_seed}-{stamp}-pid{pid}.prof"
        )
    else:
        path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats(str(path))
    return str(path)


def _build_torch_profiler_context() -> AbstractContextManager[cProfile.Profile | None]:
    return nullcontext(None)


def _log_stage0_debug(
    sink: SolverDebugSink,
    request: SolverStageRequest,
    tree: PublicTree,
    table: DenseInfosetTable,
    board: Board | None,
    step: int,
) -> None:
    sink.add_scalar("stage0/tree_nodes", float(tree.node_count), step)
    sink.add_scalar("stage0/infosets", float(table.infoset_count), step)
    sink.add_scalar("stage0/leaf_nodes", float(tree.node_count - len(table.node_to_infoset)), step)
    sink.add_scalar("stage0/depth_limit", float(request.depth_limit), step)
    sink.add_text("stage0/board", str(board) if board is not None else "none", step)
    sink.add_scalar("stage0/node_to_infoset_nonempty", float(sum(1 for value in table.node_to_infoset if value >= 0)), step)
    sink.add_scalar("stage0/root_child_count", float(tree.child_count[0] if tree.child_count else 0), step)
    log_text_map(
        sink,
        "stage0/table",
        {
            "infoset_order": table.infoset_order[:32],
            "node_to_infoset": table.node_to_infoset[:64],
            "action_counts": table.action_counts[:64],
        },
        step,
        limit=8,
    )


def _log_stage1_debug(sink: SolverDebugSink, forward: ForwardProfileResult, step: int) -> None:
    log_summary(sink, "stage1/node_reach", forward.node_reach, step)
    log_summary(sink, "stage1/infoset_reach", forward.infoset_reach, step)
    log_text_map(
        sink,
        "stage1/strategy",
        {
            "action_reach": forward.action_reach[:32],
        },
        step,
        limit=8,
    )


def _log_stage2_debug(sink: SolverDebugSink, aggregate: AggregateProbSumResult, step: int) -> None:
    log_summary(sink, "stage2/node_reach", aggregate.node_aggregate.reach, step)
    log_summary(sink, "stage2/card_reach", aggregate.node_aggregate.card_reach.ravel(), step)
    log_summary(sink, "stage2/hand_reach", aggregate.node_aggregate.hand_reach.ravel(), step)
    log_summary(sink, "stage2/leaf_reach_sum", aggregate.leaf_reach_sum, step)
    log_summary(sink, "stage2/leaf_batch_features", aggregate.leaf_batch.features.ravel(), step)
    sink.add_text("stage2/leaf_node_ids", str(aggregate.leaf_node_ids[:64]), step)


def _log_stage3_stage4_debug(
    sink: SolverDebugSink,
    showdown: ShowdownEquityResult,
    opponent_reach: OpponentReachResult,
    leaf_values: tuple[float, ...],
    step: int,
) -> None:
    log_summary(sink, "stage3/opponent_reach", opponent_reach.node_opponent_reach, step)
    log_summary(sink, "stage3/opponent_share", opponent_reach.node_opponent_share, step)
    log_summary(sink, "stage3/hand_opponent_reach", opponent_reach.node_hand_opponent_reach.ravel(), step)
    log_summary(sink, "stage4/showdown", showdown.node_showdown_equity, step)
    log_summary(sink, "stage4/showdown_bb", showdown.node_showdown_equity_bb, step)
    sink.add_histogram("stage5/leaf_values", np.asarray(leaf_values, dtype=np.float64), step)
    sink.add_sample("stage5/leaf_values_sample", leaf_values, step, limit=32)


def _log_stage6_debug(
    sink: SolverDebugSink,
    backward: BackwardCFVResult,
    leaf_values: tuple[float, ...],
    step: int,
) -> None:
    log_summary(sink, "stage6/node_cfv", backward.node_values, step)
    log_summary(sink, "stage6/infoset_cfv", backward.infoset_values, step)
    log_summary(sink, "stage6/action_values", [value for row in backward.action_values for value in row], step)
    sink.add_histogram("stage6/leaf_values", np.asarray(leaf_values, dtype=np.float64), step)


def _log_stage7_debug(
    sink: SolverDebugSink,
    final_state: DenseCfrState | None,
    dense_state: DenseCfrState | None,
    table: DenseInfosetTable,
    step: int,
) -> None:
    if final_state is None:
        return
    root_infoset = table.infoset_order[0] if table.infoset_order else None
    log_text_map(
        sink,
        "stage7/state",
        {
            "regret_rows": len(final_state.regret_sums),
            "strategy_rows": len(final_state.strategy_sums),
            "infosets": table.infoset_count,
            "root_infoset": root_infoset if root_infoset is not None else "none",
        },
        step,
        limit=8,
    )
    flat_regrets = [value for row in final_state.regret_sums for value in row]
    flat_strategy = [value for row in final_state.strategy_sums for value in row]
    log_summary(sink, "stage7/regret_sums", flat_regrets, step)
    log_summary(sink, "stage7/strategy_sums", flat_strategy, step)
    if root_infoset is not None:
        log_summary(sink, "stage7/root_regrets", final_state.regret_sums[root_infoset], step)
        log_summary(sink, "stage7/root_strategy_sums", final_state.strategy_sums[root_infoset], step)
    if dense_state is not None:
        delta_regrets = [
            new - old
            for new_row, old_row in zip(final_state.regret_sums, dense_state.regret_sums, strict=True)
            for new, old in zip(new_row, old_row, strict=True)
        ]
        delta_strategy = [
            new - old
            for new_row, old_row in zip(final_state.strategy_sums, dense_state.strategy_sums, strict=True)
            for new, old in zip(new_row, old_row, strict=True)
        ]
        log_summary(sink, "stage7/regret_delta", delta_regrets, step)
        log_summary(sink, "stage7/strategy_delta", delta_strategy, step)
        if root_infoset is not None:
            log_summary(sink, "stage7/root_regret_delta", [
                new - old
                for new, old in zip(final_state.regret_sums[root_infoset], dense_state.regret_sums[root_infoset], strict=True)
            ], step)
            log_summary(sink, "stage7/root_strategy_delta", [
                new - old
                for new, old in zip(final_state.strategy_sums[root_infoset], dense_state.strategy_sums[root_infoset], strict=True)
            ], step)


def _tree_debug_fields(tree: PublicTree, table: DenseInfosetTable) -> dict[str, object]:
    leaf_count = sum(1 for node_type in tree.node_types if node_type is NodeType.LEAF)
    depth_limit_hit_count = sum(
        1
        for node_type, payoff in zip(tree.node_types, tree.terminal_payoffs, strict=True)
        if node_type is NodeType.LEAF and payoff is None
    )
    terminal_count = sum(1 for node_type in tree.node_types if node_type is NodeType.TERMINAL)
    chance_count = sum(1 for node_type in tree.node_types if node_type is NodeType.CHANCE)
    player_count = sum(1 for node_type in tree.node_types if node_type in {NodeType.PLAYER0, NodeType.PLAYER1})
    branch_counts = [count for count in tree.child_count if count > 0]
    max_branching = max(branch_counts) if branch_counts else 0
    avg_branching = (sum(branch_counts) / len(branch_counts)) if branch_counts else 0.0
    return {
        "tree_nodes": tree.node_count,
        "tree_player_nodes": player_count,
        "tree_chance_nodes": chance_count,
        "tree_terminal_nodes": terminal_count,
        "tree_leaf_nodes": leaf_count,
        "tree_depth_limit_hits": depth_limit_hit_count,
        "tree_infosets": table.infoset_count,
        "tree_max_branching": max_branching,
        "tree_avg_branching": avg_branching,
        "tree_root_child_count": tree.child_count[0] if tree.child_count else 0,
    }
