from __future__ import annotations

import cProfile
import pstats
from dataclasses import dataclass
from io import StringIO
from time import perf_counter

from pokergpu.cfr.solver import evaluate_showdown_node_values
from pokergpu.cfr.stage1 import ForwardProfileResult, propagate_forward
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
    forward: ForwardProfileResult


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


def run_profile_workload(ctx: ProfileContext) -> None:
    aggregate = aggregate_prob_sum(ctx.tree, ctx.forward, ctx.board)
    opponent = compute_opponent_reach(ctx.tree, aggregate)
    cache = build_showdown_equity_board_cache(ctx.board)
    showdown_input = build_showdown_equity_input(ctx.tree, aggregate, opponent, board=ctx.board)

    compute_showdown_equity(ctx.tree, aggregate, opponent, board=ctx.board)
    compute_showdown_equity_node(showdown_input.rows[0], cache=cache)
    evaluate_showdown_node_values(ctx.tree, ctx.forward, board=ctx.board)


def profile_once() -> str:
    ctx = make_profile_context()
    profiler = cProfile.Profile()
    total_start = perf_counter()
    profiler.enable()
    run_profile_workload(ctx)
    profiler.disable()
    total_seconds = perf_counter() - total_start

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(40)
    stats.print_callers(20)
    return f"total_seconds={total_seconds:.6f}\n{stream.getvalue()}"


def main() -> None:
    print(profile_once())


if __name__ == "__main__":
    main()
