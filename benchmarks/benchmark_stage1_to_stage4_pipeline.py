from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from pokergpu.cfr.solver import evaluate_showdown_node_values
from pokergpu.cfr.stage1 import ForwardProfileResult, propagate_forward
from pokergpu.cfr.stage2 import AggregateProbSumResult, aggregate_prob_sum
from pokergpu.cfr.stage3 import OpponentReachResult, compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityResult, compute_showdown_equity
from pokergpu.core.board import Board
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


@dataclass(frozen=True, slots=True)
class StageTiming:
    name: str
    warmup_seconds: float
    measured_seconds: float
    runs: int

    @property
    def average_ms(self) -> float:
        return (self.measured_seconds / self.runs) * 1000.0 if self.runs > 0 else 0.0


def make_profile_tree() -> PublicTree:
    return PublicTree(
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


def run_stage_1(tree: PublicTree) -> ForwardProfileResult:
    return propagate_forward(tree)


def run_stage_2(tree: PublicTree, forward: ForwardProfileResult) -> AggregateProbSumResult:
    return aggregate_prob_sum(tree, forward)


def run_stage_3(tree: PublicTree, aggregate: AggregateProbSumResult) -> OpponentReachResult:
    return compute_opponent_reach(tree, aggregate)


def run_stage_4(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    board: Board,
) -> ShowdownEquityResult:
    return compute_showdown_equity(tree, aggregate, opponent_reach, board=board)


def profile_stage(name: str, runs: int, fn: Callable[[], object]) -> StageTiming:
    warmup_start = perf_counter()
    fn()
    warmup_seconds = perf_counter() - warmup_start

    measured_start = perf_counter()
    for _ in range(runs):
        fn()
    measured_seconds = perf_counter() - measured_start
    return StageTiming(
        name=name,
        warmup_seconds=warmup_seconds,
        measured_seconds=measured_seconds,
        runs=runs,
    )


def main() -> None:
    tree = make_profile_tree()
    board = Board.from_str("AhKdTc9s2c")

    forward = run_stage_1(tree)
    aggregate = run_stage_2(tree, forward)
    opponent_reach = run_stage_3(tree, aggregate)

    timings = [
        profile_stage("stage1_forward", 5000, lambda: run_stage_1(tree)),
        profile_stage("stage2_aggregate", 2000, lambda: run_stage_2(tree, forward)),
        profile_stage("stage3_opponent", 2000, lambda: run_stage_3(tree, aggregate)),
        profile_stage("stage4_showdown_cache+node", 200, lambda: run_stage_4(tree, aggregate, opponent_reach, board)),
        profile_stage("stage4_node_only", 2000, lambda: evaluate_showdown_node_values(tree, forward, board=board)),
    ]

    print(f"tree_nodes={tree.node_count} board={board}")
    for timing in timings:
        print(
            f"{timing.name}: runs={timing.runs} "
            f"warmup_ms={timing.warmup_seconds * 1000.0:.3f} "
            f"avg_ms={timing.average_ms:.3f} "
            f"total_ms={timing.measured_seconds * 1000.0:.3f}"
        )


if __name__ == "__main__":
    main()
