from __future__ import annotations

from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import cProfile
import pstats
import time
from types import TracebackType

import numpy as np

from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
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
from pokergpu.tree.public_tree import PublicTree

from .evaluation import evaluate_leaf_node_values
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
        backend: GpuLeafBackend | None = None,
        infoset_strategies: dict[InfosetId, tuple[float, ...]] | None = None,
        max_workers: int | None = None,
        executor: Executor | None = None,
    ) -> SolverStageResult:
        assert tree.node_count > 0, "public tree cannot be empty"
        profiler_output = None
        profiler_cm = _build_profiler_context(request)
        with profiler_cm as profiler:
            timings: dict[str, float] = {}
            total_start = time.perf_counter()

            stage_start = time.perf_counter()
            table = build_dense_infoset_table(tree)
            timings["stage0_table"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            forward = propagate_forward(
                tree,
                infoset_strategies=infoset_strategies,
            )
            timings["stage1_forward"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            aggregate = aggregate_prob_sum(tree, forward, board, max_workers=max_workers)
            timings["stage2_aggregate"] = time.perf_counter() - stage_start

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
                "tree_nodes": tree.node_count,
            },
        )


def run_solver_stage(
    request: SolverStageRequest,
    *,
    tree: PublicTree,
    dense_state: DenseCfrState | None = None,
    board: Board | None = None,
    backend: GpuLeafBackend | None = None,
    infoset_strategies: dict[InfosetId, tuple[float, ...]] | None = None,
    max_workers: int | None = None,
    executor: Executor | None = None,
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
    backend: GpuLeafBackend | None,
    max_workers: int | None,
) -> tuple[float, ...]:
    _ = board
    leaf_result = evaluate_leaf_node_values(
        tree,
        forward,
        board=None,
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
        return None
    path = Path(output_path)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats(str(path))
    return str(path)


def _build_torch_profiler_context() -> AbstractContextManager[cProfile.Profile | None]:
    return nullcontext(None)
