from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from pokergpu.cfr.solver import evaluate_showdown_node_values
from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import build_showdown_equity_board_cache
from pokergpu.cfr.stage4 import build_showdown_equity_input
from pokergpu.cfr.stage4 import compute_showdown_equity
from pokergpu.cfr.stage4 import compute_showdown_equity_node
from pokergpu.core.board import Board
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


@dataclass(frozen=True, slots=True)
class ProfileContext:
    tree: PublicTree
    board: Board
    forward: object


def make_profile_context() -> ProfileContext:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.LEAF,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 3, 3),
        child_count=(2, 1, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), None, None),
        terminal_payoffs=(None, None, None, Chips(2)),
    )
    board = Board.from_str("AhKdTc9s2c")
    forward = propagate_forward(tree)
    return ProfileContext(tree=tree, board=board, forward=forward)


def run_workload(ctx: ProfileContext) -> None:
    aggregate = aggregate_prob_sum(ctx.tree, ctx.forward, ctx.board)
    opponent = compute_opponent_reach(ctx.tree, aggregate)
    cache = build_showdown_equity_board_cache(ctx.board)
    showdown_input = build_showdown_equity_input(ctx.tree, aggregate, opponent, board=ctx.board)

    compute_showdown_equity(ctx.tree, aggregate, opponent, board=ctx.board)
    compute_showdown_equity_node(showdown_input.rows[0], cache=cache)
    evaluate_showdown_node_values(ctx.tree, ctx.forward, board=ctx.board)


def main() -> None:
    ctx = make_profile_context()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        total_start = perf_counter()
        with torch.profiler.record_function("stage4_workload"):
            run_workload(ctx)
        total_seconds = perf_counter() - total_start

    print(f"total_seconds={total_seconds:.6f}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
