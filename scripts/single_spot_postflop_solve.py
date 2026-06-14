from __future__ import annotations

import argparse
import json
import os
import time
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.abstraction.hands import RangeVector
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.cards import Card, card_from_str
from pokergpu.core.state import GameState, PlayerState
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.runtime.gpu_postflop import _prepare_gpu_solve, _run_gpu_solve
from pokergpu.runtime.postflop import PostflopResolveSpec

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def main() -> None:
    args = _parse_args()
    if args.cuda_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.cuda_dsa:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
    state = GameState(
        board=Board.from_str(args.board),
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=_parse_hole_cards(args.hole_cards_p0)),
            PlayerState(player=PlayerIndex(1), hole_cards=_parse_hole_cards(args.hole_cards_p1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(args.pot)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(args.stack0)),
                PlayerStack(player=PlayerIndex(1), stack=chips(args.stack1)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(
                small_blind=chips(args.small_blind),
                big_blind=chips(args.big_blind),
            ),
            to_act=PlayerIndex(args.to_act),
        ),
        dealer=PlayerIndex(args.dealer),
    )
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=args.time_budget_sec,
        iterations=args.iterations,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        min_reach_prob=args.min_reach_prob,
    )

    evaluator = CpuStubLeafEvaluator()
    started_at = time.monotonic()
    packed = _prepare_gpu_solve(spec)
    warmup_trace = _run_gpu_solve(packed, evaluator, debug=args.debug)
    if args.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
        ) as prof:
            with record_function("solve::run_gpu_solve"):
                trace = _run_gpu_solve(packed, evaluator, debug=args.debug)
        _print_profiler_tables(prof)
    else:
        trace = warmup_trace
    total_seconds = time.monotonic() - started_at

    print(
        json.dumps(
            {
                "spot": {
                    "board": args.board,
                    "pot": args.pot,
                    "stacks": [args.stack0, args.stack1],
                    "to_act": args.to_act,
                    "dealer": args.dealer,
                },
                "tree": {
                    "node_count": packed.tree.tree.node_count,
                    "leaf_count": int(packed.plan.frontier_nodes.numel()),
                    "level_widths": [len(level) for level in packed.tree.level_schedule.level_nodes],
                    "level_node_counts": list(trace.level_node_counts),
                    "level_edge_counts": list(trace.level_edge_counts),
                    "level_frontier_counts": list(trace.level_frontier_counts),
                    "compact_forward_level_sizes": list(trace.compact_forward_level_sizes),
                    "compact_backward_level_sizes": list(trace.compact_backward_level_sizes),
                    "frontier_batch_size": int(packed.plan.frontier_nodes.numel()),
                },
                "timing": {
                    "elapsed_seconds": trace.elapsed_seconds,
                    "total_seconds": total_seconds,
                    "iterations": trace.iterations,
                    "phase_seconds": trace.phase_seconds,
                },
                "result": {
                    "root_infoset_id": trace.packed.root_infoset,
                    "root_actions": trace.packed.root_actions,
                    "root_strategy": trace.root_strategy.tolist(),
                    "root_action_ev_player0": trace.root_action_ev_player0.tolist(),
                    "root_action_ev_player1": trace.root_action_ev_player1.tolist(),
                    "root_ev_player0": trace.root_ev_player0,
                    "root_ev_player1": trace.root_ev_player1,
                },
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve one postflop spot on GPU")
    parser.add_argument("--board", default="AhKdTc")
    parser.add_argument("--pot", type=int, default=300)
    parser.add_argument("--stack0", type=int, default=1700)
    parser.add_argument("--stack1", type=int, default=1700)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--to-act", type=int, default=0)
    parser.add_argument("--dealer", type=int, default=0)
    parser.add_argument("--hole-cards-p0", default="")
    parser.add_argument("--hole-cards-p1", default="")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--time-budget-sec", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--min-reach-prob", type=float, default=0.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda-blocking", action="store_true")
    parser.add_argument("--cuda-dsa", action="store_true")
    return parser.parse_args()


def _parse_hole_cards(value: str) -> tuple[Card, ...] | None:
    text = value.strip()
    if not text:
        return None
    if len(text) != 4:
        raise ValueError("hole cards must be 4 characters like AsKs")
    return (card_from_str(text[:2]), card_from_str(text[2:]))


def _print_profiler_tables(prof: object) -> None:
    print("== CPU ops ==")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print("== CUDA kernels ==")
    print(_format_cuda_kernel_table(prof))


def _format_cuda_kernel_table(prof: object, row_limit: int = 50) -> str:
    events = getattr(prof, "events", lambda: [])()
    stats: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "cuda_time_total_us": 0.0, "cpu_time_total_us": 0.0})
    for event in events:
        device_type = getattr(event, "device_type", None)
        device_name = getattr(device_type, "name", str(device_type))
        if device_name != "CUDA":
            continue
        name = str(getattr(event, "name", "unknown"))
        stats[name]["count"] += 1
        stats[name]["cuda_time_total_us"] += float(getattr(event, "cuda_time_total", 0.0))
        stats[name]["cpu_time_total_us"] += float(getattr(event, "cpu_time_total", 0.0))
    if not stats:
        return "No CUDA kernel events recorded."
    rows = sorted(stats.items(), key=lambda item: float(item[1]["cuda_time_total_us"]), reverse=True)[:row_limit]
    lines = [
        f"{'Name':60}  {'Count':>7}  {'CUDA total (us)':>15}  {'CPU total (us)':>14}",
        "-" * 105,
    ]
    for name, data in rows:
        lines.append(
            f"{name[:60]:60}  {int(data['count']):7d}  {float(data['cuda_time_total_us']):15.3f}  {float(data['cpu_time_total_us']):14.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
