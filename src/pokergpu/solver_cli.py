from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
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
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.cfr.solver import run_solver_stage
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.cfr.solver.state import DenseCfrState as SolverDenseCfrState
from pokergpu.cfr.solver.tree import resolve_game_state_spec
from pokergpu.core import BettingRoundState
from pokergpu.core import BlindStructure
from pokergpu.core import GameState
from pokergpu.core import HandPhase
from pokergpu.core import PlayerBet
from pokergpu.core import PlayerIndex
from pokergpu.core import PlayerStack
from pokergpu.core import Pot
from pokergpu.core.board import Board
from pokergpu.core.cards import shuffled_deck
from pokergpu.core.cards import Card
from pokergpu.core.betting import chips
from pokergpu.core.signatures import public_state_signature
from pokergpu.core.state import PlayerState
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
    parser.add_argument("--debug", action="store_true", help="print detailed solver debug information")
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
    board = _resolve_board(args, request)
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
    _print_summary(
        result,
        current if isinstance(current, SolverDenseCfrState) else None,
        tree,
        board,
        debug=args.debug,
    )
    if args.summary_output is not None:
        _write_summary(
            args.summary_output,
            result,
            current if isinstance(current, SolverDenseCfrState) else None,
            tree,
            board,
        )
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


def _resolve_board(args: argparse.Namespace, request: SolverStageRequest) -> Board:
    if args.board:
        return Board.from_str(args.board)
    if request.state is not None and request.state.seed is not None:
        rng = Random(request.state.seed)
        deck = shuffled_deck(rng)
        if request.game is GameVariant.LEDUC:
            return Board(cards=tuple(deck[:0]))
        if request.game in {GameVariant.HOLDEM_HU, GameVariant.HOLDEM_6MAX}:
            return Board(cards=tuple(deck[:0]))
    return Board(cards=())


def _make_dense_state(tree: PublicTree) -> DenseCfrState:
    from pokergpu.cfr.solver import build_dense_infoset_table

    table = build_dense_infoset_table(tree)
    return DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )


def _print_summary(
    result: SolverStageResult,
    dense_state: DenseCfrState | None,
    tree: PublicTree,
    board: Board,
    *,
    debug: bool = False,
) -> None:
    print(f"game={result.request.game.value}")
    print(f"variant={result.request.cfr_variant.value}")
    print(f"depth={result.request.depth_limit}")
    print(f"seed={result.request.seed if result.request.seed is not None else 'none'}")
    if debug:
        _print_debug_details(result, board, tree, dense_state)
    if dense_state is not None:
        root_strategy = _format_root_strategy(dense_state, tree)
        if root_strategy is not None:
            print(f"root_strategy={root_strategy}")
    if result.timing_seconds is not None:
        total = result.timing_seconds.get("total", 0.0)
        print(f"total_seconds={total:.6f}")
    if result.profiler_output is not None:
        print(f"profile_output={result.profiler_output}")


def _write_summary(
    path: Path,
    result: SolverStageResult,
    dense_state: DenseCfrState | None,
    tree: PublicTree,
    board: Board,
) -> None:
    payload = {
        "game": result.request.game.value,
        "variant": result.request.cfr_variant.value,
        "depth": result.request.depth_limit,
        "seed": result.request.seed,
        "board": str(board),
        "timing_seconds": result.timing_seconds,
        "diagnostics": result.diagnostics,
        "profiler_output": result.profiler_output,
        "root_strategy": _format_root_strategy(dense_state, tree),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_debug_details(
    result: SolverStageResult,
    board: Board,
    tree: PublicTree,
    dense_state: DenseCfrState | None,
) -> None:
    print(f"board={board if str(board) else 'preflop'}")
    print(f"board_cards={_format_board_cards(board)}")
    print(f"state_mode={result.request.state.mode.value if result.request.state is not None else 'none'}")
    decoded_state = _decode_debug_game_state(result)
    if decoded_state is not None:
        print(f"public_state={public_state_signature(decoded_state)}")
        print(f"chips={_format_chips(decoded_state)}")
        print(f"player_cards={_format_player_cards(decoded_state)}")
    else:
        print("chips=unavailable")
        print("player_cards=unavailable")
    if result.request.state is not None:
        print(f"state_seed={result.request.state.seed if result.request.state.seed is not None else 'none'}")
        if result.request.state.encoded_state is not None:
            print(f"encoded_state={result.request.state.encoded_state.hex()}")
            print(f"encoded_state_bytes={len(result.request.state.encoded_state)}")
    print(f"iterations={result.request.iterations}")
    print(f"cpu_workers={result.request.cpu_workers}")
    print(f"cpu_workers_stage3={result.request.effective_cpu_workers_stage3}")
    print(f"cpu_workers_stage4={result.request.effective_cpu_workers_stage4}")
    print(f"cpu_workers_stage6={result.request.effective_cpu_workers_stage6}")
    print(f"cpu_workers_stage7={result.request.effective_cpu_workers_stage7}")
    diagnostics = result.diagnostics or {}
    if diagnostics:
        for key in sorted(diagnostics):
            print(f"diagnostic.{key}={diagnostics[key]}")
    if dense_state is not None:
        root_strategy = _format_root_strategy(dense_state, tree)
        if root_strategy is not None:
            print(f"debug.root_strategy={root_strategy}")


def _format_board_cards(board: Board) -> str:
    if not board.cards:
        return "[]"
    return "[" + ", ".join(str(card) for card in board.cards) + "]"


def _decode_debug_game_state(result: SolverStageResult) -> GameState | None:
    state_spec = result.request.state
    if state_spec is None or state_spec.encoded_state is None:
        return None
    try:
        payload = json.loads(state_spec.encoded_state.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        board = Board.from_str(str(payload.get("board", "")))
        players = tuple(
            PlayerState(
                player=PlayerIndex(int(player_payload["player"])),
                hole_cards=_parse_hole_cards(player_payload.get("hole_cards")),
                folded=bool(player_payload.get("folded", False)),
                all_in=bool(player_payload.get("all_in", False)),
            )
            for player_payload in payload.get("players", [])
        )
        stacks = tuple(
            PlayerStack(
                player=PlayerIndex(int(stack_payload["player"])),
                stack=chips(int(stack_payload["stack"])),
            )
            for stack_payload in payload.get("stacks", [])
        )
        bets = tuple(
            PlayerBet(
                player=PlayerIndex(int(bet_payload["player"])),
                committed=chips(int(bet_payload.get("committed", 0))),
                folded=bool(bet_payload.get("folded", False)),
                all_in=bool(bet_payload.get("all_in", False)),
            )
            for bet_payload in payload.get("bets", [])
        )
        blinds_payload = payload.get("blinds", {})
        blinds = BlindStructure(
            small_blind=chips(int(blinds_payload.get("small_blind", 1))),
            big_blind=chips(int(blinds_payload.get("big_blind", 2))),
            ante=chips(int(blinds_payload.get("ante", 0))),
        )
        betting_round = BettingRoundState(
            pot=Pot(amount=chips(int(payload.get("pot", 0)))),
            stacks=stacks,
            bets=bets,
            blinds=blinds,
            to_act=PlayerIndex(int(payload.get("to_act", 0))),
        )
        return GameState(
            board=board,
            players=players,
            betting_round=betting_round,
            phase=HandPhase(str(payload.get("phase", HandPhase.IN_PROGRESS.value))),
            dealer=PlayerIndex(int(payload.get("dealer", 0))),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_hole_cards(value: object) -> tuple[Card, Card] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    first = Card.from_str(str(value[0]))
    second = Card.from_str(str(value[1]))
    return (first, second)


def _format_chips(state: GameState) -> str:
    parts: list[str] = []
    for player in state.players:
        stack = next(stack.stack for stack in state.betting_round.stacks if stack.player == player.player)
        parts.append(f"p{int(player.player)}={int(stack)}")
    return "[" + ", ".join(parts) + "]"


def _format_player_cards(state: GameState) -> str:
    parts: list[str] = []
    for player in state.players:
        if player.hole_cards is None:
            parts.append(f"p{int(player.player)}=none")
        else:
            parts.append(f"p{int(player.player)}={player.hole_cards[0]}{player.hole_cards[1]}")
    return "[" + ", ".join(parts) + "]"


def _format_root_strategy(dense_state: DenseCfrState | None, tree: PublicTree) -> str | None:
    if dense_state is None:
        return None
    table = build_dense_infoset_table(tree)
    if not table.infoset_order:
        return None
    root_infoset = table.infoset_order[0]
    strategy_sums = dense_state.strategy_sums[root_infoset]
    if not strategy_sums:
        return None
    total = sum(max(0.0, value) for value in strategy_sums)
    if total <= 0.0:
        strategy = tuple(1.0 / len(strategy_sums) for _ in strategy_sums)
    else:
        strategy = tuple(max(0.0, value) / total for value in strategy_sums)
    return "(" + ", ".join(f"{value:.3f}" for value in strategy) + ")"
