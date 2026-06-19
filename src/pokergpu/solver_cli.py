from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import ModuleType
from typing import Iterable
from typing import cast

from pokergpu.cfr.solver import CfrVariant
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import GameStateMode
from pokergpu.cfr.solver import GameStateSpec
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import ProfilingKind
from pokergpu.cfr.solver import ProfilerSpec
from pokergpu.cfr.solver import SolverStageRequest
from pokergpu.cfr.solver import SolverStageResult
from pokergpu.cfr.solver import TimingSpec
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.cfr.solver.state import DenseCfrState as SolverDenseCfrState
from pokergpu.cfr.solver.tree import resolve_game_state_spec
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import PublicTree

try:
    import tqdm as tqdm_module
except ImportError:  # pragma: no cover
    class _TqdmModule:
        @staticmethod
        def tqdm(
            iterable: Iterable[int],
            total: int | None = None,
            desc: str | None = None,
        ) -> Iterable[int]:
            _ = total, desc
            return iterable

    tqdm_module = cast(ModuleType, _TqdmModule())


def _progress_bar(
    iterable: Iterable[int],
    total: int | None = None,
    desc: str | None = None,
) -> Iterable[int]:
    return tqdm_module.tqdm(iterable, total=total, desc=desc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokergpu-solver", description="Run a CFR solver stage")
    parser.add_argument("--game", choices=[variant.value for variant in GameVariant], required=True)
    parser.add_argument("--variant", choices=[variant.value for variant in CfrVariant], required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--cpu-workers-stage3", type=int)
    parser.add_argument("--cpu-workers-stage4", type=int)
    parser.add_argument("--cpu-workers-stage6", type=int)
    parser.add_argument("--cpu-workers-stage7", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--state-mode", choices=[mode.value for mode in GameStateMode], default="random")
    parser.add_argument("--encoded-state", type=str)
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument("--profile", choices=[kind.value for kind in ProfilingKind])
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--board", type=str)
    parser.add_argument("--progress", action="store_true", help="show a tqdm progress bar")
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
        cpu_workers_stage3=args.cpu_workers_stage3,
        cpu_workers_stage4=args.cpu_workers_stage4,
        cpu_workers_stage6=args.cpu_workers_stage6,
        cpu_workers_stage7=args.cpu_workers_stage7,
        profiler=_build_profiler_spec(args),
        timing=TimingSpec(measure=args.measure_time),
        measure_time=args.measure_time,
    )

    tree = make_game_public_tree(request.game)
    board = Board.from_str(args.board) if args.board else Board(cards=())
    dense_state = _make_dense_state(tree)

    current = dense_state
    iterator: Iterable[int] = range(request.iterations)
    if args.progress and request.iterations > 1:
        iterator = _progress_bar(iterator, total=request.iterations, desc="solver")

    result = None
    for _ in iterator:
        result = run_solver_stage(request, tree=tree, dense_state=current, board=board)
        if isinstance(result.final_state, SolverDenseCfrState):
            current = result.final_state

    assert result is not None
    _print_summary(result)
    if args.summary_output is not None:
        _write_summary(args.summary_output, result)
    return 0


def _build_state_spec(args: argparse.Namespace) -> GameStateSpec | None:
    if args.state_mode == GameStateMode.EXACT.value:
        encoded = args.encoded_state.encode() if args.encoded_state is not None else b""
        return GameStateSpec(mode=GameStateMode.EXACT, encoded_state=encoded, seed=args.seed)
    if args.state_mode == GameStateMode.RANDOM.value:
        return GameStateSpec(mode=GameStateMode.RANDOM, seed=args.seed)
    return resolve_game_state_spec(None)


def _build_profiler_spec(args: argparse.Namespace) -> ProfilerSpec | None:
    if args.profile is None:
        return None
    return ProfilerSpec(
        kind=ProfilingKind(args.profile),
        output_path=str(args.profile_output) if args.profile_output is not None else None,
    )


def _make_dense_state(tree: PublicTree) -> DenseCfrState:
    from pokergpu.cfr.solver import build_dense_infoset_table

    table = build_dense_infoset_table(tree)
    return DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )


def _print_summary(result: SolverStageResult) -> None:
    print(f"game={result.request.game.value}")
    print(f"variant={result.request.cfr_variant.value}")
    print(f"depth={result.request.depth_limit}")
    if result.timing_seconds is not None:
        total = result.timing_seconds.get("total", 0.0)
        print(f"total_seconds={total:.6f}")
    if result.profiler_output is not None:
        print(f"profile_output={result.profiler_output}")


def _write_summary(path: Path, result: SolverStageResult) -> None:
    payload = {
        "game": result.request.game.value,
        "variant": result.request.cfr_variant.value,
        "depth": result.request.depth_limit,
        "timing_seconds": result.timing_seconds,
        "diagnostics": result.diagnostics,
        "profiler_output": result.profiler_output,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
