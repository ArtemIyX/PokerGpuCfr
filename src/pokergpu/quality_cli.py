from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pokergpu.cfr.leaf_backend_factory import create_heuristic_leaf_backend
from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import GameStateMode
from pokergpu.cfr.solver import GameStateSpec
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import PublicTree
from pokergpu.solver_cli import _build_state_spec
from pokergpu.solver_cli import _dense_state_to_infoset_strategies
from pokergpu.solver_cli import _make_dense_state
from pokergpu.solver_cli import _resolve_board


@dataclass(slots=True, frozen=True)
class SolverQualityReport:
    iteration: int
    game: str
    variant: str
    root_strategy: tuple[float, ...]
    root_entropy: float
    avg_entropy: float
    mean_abs_regret: float
    max_abs_regret: float
    positive_regret_sum: float
    root_strategy_delta: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokergpu-quality", description="Evaluate solver quality")
    parser.add_argument("--game", choices=[variant.value for variant in GameVariant], required=True)
    parser.add_argument("--variant", choices=[variant.value for variant in CfrVariant], required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--state-mode", choices=[mode.value for mode in GameStateMode], default="random")
    parser.add_argument("--encoded-state", type=str)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--board", type=str)
    parser.add_argument("--leaf-evaluator", choices=["heuristic", "model", "triton"], default="heuristic")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--trace-iterations", action="store_true", help="print a table at iterations 1,5,10,20,50,100")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    request = SolverStageRequest(
        game=GameVariant(args.game),
        cfr_variant=CfrVariant(args.variant),
        depth_limit=args.depth,
        iterations=args.iterations,
        state=_build_state_spec(args),
        seed=args.seed,
        cpu_workers=args.cpu_workers,
        measure_time=False,
    )
    tree = make_game_public_tree(request.game)
    table = build_dense_infoset_table(tree)
    board = _resolve_board(args, request)
    dense_state = _make_dense_state(tree)
    backend = create_heuristic_leaf_backend()

    current = dense_state
    result = None
    trace_checkpoints = (1, 5, 10, 20, 50, 100)
    trace_reports: list[SolverQualityReport] = []
    previous_trace_root_strategy: tuple[float, ...] | None = None
    for iteration_index in range(1, request.iterations + 1):
        infoset_strategies = _dense_state_to_infoset_strategies(current, tree)
        result = run_solver_stage(
            request,
            tree=tree,
            dense_state=current,
            board=board,
            backend=backend,
            infoset_strategies=infoset_strategies,
        )
        if isinstance(result.final_state, DenseCfrState):
            current = result.final_state
        if args.trace_iterations and iteration_index in trace_checkpoints:
            current_root_strategy = _normalize(current.strategy_sums[table.infoset_order[0]])
            trace_reports.append(
                build_quality_report(
                    current,
                    tree,
                    request,
                    iteration=iteration_index,
                    previous_root_strategy=previous_trace_root_strategy,
                )
            )
            previous_trace_root_strategy = current_root_strategy

    assert result is not None
    report = build_quality_report(
        current,
        tree,
        request,
        iteration=request.iterations,
        previous_root_strategy=previous_trace_root_strategy,
    )
    _print_report(report)
    if args.trace_iterations:
        _print_trace_table(trace_reports)
    if args.summary_output is not None:
        args.summary_output.write_text(_format_report_json(report), encoding="utf-8")
    return 0


def build_quality_report(
    state: DenseCfrState,
    tree: PublicTree,
    request: SolverStageRequest,
    *,
    iteration: int,
    previous_root_strategy: tuple[float, ...] | None = None,
) -> SolverQualityReport:
    table = build_dense_infoset_table(tree)
    root_infoset = table.infoset_order[0]
    root_strategy = _normalize(state.strategy_sums[root_infoset])
    root_entropy = _entropy(root_strategy)
    entropies = [_entropy(_normalize(row)) for row in state.strategy_sums if sum(row) > 0.0]
    flat_regrets = [value for row in state.regret_sums for value in row]
    mean_abs_regret = sum(abs(value) for value in flat_regrets) / len(flat_regrets) if flat_regrets else 0.0
    max_abs_regret = max((abs(value) for value in flat_regrets), default=0.0)
    positive_regret_sum = sum(max(0.0, value) for value in flat_regrets)
    root_strategy_delta = 0.0
    if previous_root_strategy is not None and len(previous_root_strategy) == len(root_strategy):
        root_strategy_delta = sum(abs(a - b) for a, b in zip(previous_root_strategy, root_strategy))
    return SolverQualityReport(
        iteration=iteration,
        game=request.game.value,
        variant=request.cfr_variant.value,
        root_strategy=root_strategy,
        root_entropy=root_entropy,
        avg_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
        mean_abs_regret=mean_abs_regret,
        max_abs_regret=max_abs_regret,
        positive_regret_sum=positive_regret_sum,
        root_strategy_delta=root_strategy_delta,
    )


def _print_report(report: SolverQualityReport) -> None:
    print(f"game={report.game}")
    print(f"variant={report.variant}")
    print(f"iterations={report.iteration}")
    print(f"root_strategy={_format_tuple(report.root_strategy)}")
    print(f"root_entropy={report.root_entropy:.6f}")
    print(f"avg_entropy={report.avg_entropy:.6f}")
    print(f"mean_abs_regret={report.mean_abs_regret:.6f}")
    print(f"max_abs_regret={report.max_abs_regret:.6f}")
    print(f"positive_regret_sum={report.positive_regret_sum:.6f}")
    print(f"root_strategy_delta={report.root_strategy_delta:.6f}")
    print("note=positive_regret_sum is a practical exploitability proxy, not exact exploitability")


def _format_report_json(report: SolverQualityReport) -> str:
    import json

    payload = {
        "game": report.game,
        "variant": report.variant,
        "iterations": report.iteration,
        "root_strategy": report.root_strategy,
        "root_entropy": report.root_entropy,
        "avg_entropy": report.avg_entropy,
        "mean_abs_regret": report.mean_abs_regret,
        "max_abs_regret": report.max_abs_regret,
        "positive_regret_sum": report.positive_regret_sum,
        "root_strategy_delta": report.root_strategy_delta,
        "note": "positive_regret_sum is a practical exploitability proxy, not exact exploitability",
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _print_trace_table(reports: list[SolverQualityReport]) -> None:
    if not reports:
        return
    print("iteration | root_strategy | root_entropy | avg_entropy | mean_abs_regret | max_abs_regret | positive_regret_sum | root_strategy_delta")
    print("-" * 122)
    for report in reports:
        print(
            f"{report.iteration:9d} | "
            f"{_format_tuple(report.root_strategy):<13} | "
            f"{report.root_entropy:12.6f} | "
            f"{report.avg_entropy:10.6f} | "
            f"{report.mean_abs_regret:15.6f} | "
            f"{report.max_abs_regret:14.6f} | "
            f"{report.positive_regret_sum:19.6f} | "
            f"{report.root_strategy_delta:18.6f}"
        )


def _normalize(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(max(0.0, value) for value in weights)
    if total <= 0.0:
        return tuple(1.0 / len(weights) for _ in weights)
    return tuple(max(0.0, value) / total for value in weights)


def _entropy(strategy: tuple[float, ...]) -> float:
    total = 0.0
    for prob in strategy:
        if prob > 0.0:
            total -= prob * math.log(prob)
    return total


def _format_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in values) + ")"


if __name__ == "__main__":
    raise SystemExit(main())
