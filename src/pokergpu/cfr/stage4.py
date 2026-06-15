from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pokergpu.abstraction.hands import private_hand_count
from pokergpu.abstraction.hands import private_hand_mask
from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.stage3 import OpponentReachResult
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import PublicTree


@dataclass(slots=True, frozen=True)
class ShowdownEquityInputRow:
    node_id: int
    board: Board
    opponent_reach: tuple[float, ...]
    live_hand_mask: tuple[bool, ...]
    node_share: float


@dataclass(slots=True, frozen=True)
class ShowdownEquityInput:
    rows: tuple[ShowdownEquityInputRow, ...]


@dataclass(slots=True, frozen=True)
class ShowdownEquityOutputRow:
    node_id: int
    showdown_equity: float
    pot_size: float


@dataclass(slots=True, frozen=True)
class ShowdownEquityResult:
    node_showdown_equity: tuple[float, ...]
    node_showdown_equity_bb: tuple[float, ...]
    input_rows: ShowdownEquityInput
    output_rows: tuple[ShowdownEquityOutputRow, ...]


def build_showdown_equity_input(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
) -> ShowdownEquityInput:
    if tree.node_count != len(aggregate.node_aggregate.reach):
        raise ValueError("tree and aggregate result must cover the same number of nodes")
    if tree.node_count != len(opponent_reach.node_opponent_reach):
        raise ValueError("tree and opponent reach result must cover the same number of nodes")

    live_board = board or Board(())
    live_hand_mask = tuple(private_hand_mask(live_board.cards))
    rows = []
    for node_id in range(tree.node_count):
        rows.append(
            ShowdownEquityInputRow(
                node_id=node_id,
                board=live_board,
                opponent_reach=opponent_reach.node_hand_opponent_reach[node_id],
                live_hand_mask=live_hand_mask,
                node_share=opponent_reach.node_opponent_share[node_id],
            )
        )
    return ShowdownEquityInput(rows=tuple(rows))


def compute_showdown_equity(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
    max_workers: int | None = None,
) -> ShowdownEquityResult:
    showdown_input = build_showdown_equity_input(
        tree,
        aggregate,
        opponent_reach,
        board=board,
    )
    node_values = _compute_node_showdown_equity(showdown_input, max_workers=max_workers)
    return ShowdownEquityResult(
        node_showdown_equity=node_values,
        node_showdown_equity_bb=node_values,
        input_rows=showdown_input,
        output_rows=tuple(
            ShowdownEquityOutputRow(
                node_id=row.node_id,
                showdown_equity=value,
                pot_size=1.0,
            )
            for row, value in zip(showdown_input.rows, node_values, strict=True)
        ),
    )


def _compute_node_showdown_equity(
    showdown_input: ShowdownEquityInput,
    *,
    max_workers: int | None = None,
) -> tuple[float, ...]:
    if max_workers is None or max_workers <= 1 or len(showdown_input.rows) <= 1:
        return tuple(_compute_single_node_showdown_equity(row) for row in showdown_input.rows)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return tuple(executor.map(_compute_single_node_showdown_equity, showdown_input.rows))


def _compute_single_node_showdown_equity(row: ShowdownEquityInputRow) -> float:
    if len(row.opponent_reach) != private_hand_count():
        raise ValueError("opponent reach must match private hand count")
    if len(row.live_hand_mask) != private_hand_count():
        raise ValueError("live hand mask must match private hand count")
    if row.node_share < 0.0:
        raise ValueError("node share cannot be negative")
    return 0.0
