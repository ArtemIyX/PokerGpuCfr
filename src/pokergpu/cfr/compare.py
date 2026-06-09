from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .iteration import CFRVariant
from .kuhn import (
    KuhnCard,
    average_strategy_root_bet_probability,
    expected_game_value_for_average_strategy,
    train_kuhn_cfr,
)
from .leduc import (
    LeducRank,
    average_strategy_root_bet_probability_leduc,
    expected_game_value_for_average_strategy_leduc,
    train_leduc_cfr,
)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    game: str
    variant: str
    iterations: int
    avg_value_p0: float
    root_bet_j: float
    root_bet_q: float
    root_bet_k: float
    elapsed_seconds: float


def run_toy_game_comparison(
    output_path: Path,
    kuhn_iterations: Sequence[int],
    leduc_iterations: Sequence[int],
    variants: Sequence[CFRVariant],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[ComparisonRow]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = (
        [
            ("kuhn", variant, iteration)
            for variant in variants
            for iteration in kuhn_iterations
        ]
        + [
            ("leduc", variant, iteration)
            for variant in variants
            for iteration in leduc_iterations
        ]
    )
    rows: list[ComparisonRow] = []

    for index, (game, variant, iteration_count) in enumerate(tasks, start=1):
        if progress_callback is not None:
            progress_callback(
                index - 1,
                len(tasks),
                f"{game}:{variant.value}:{iteration_count}",
            )
        started_at = perf_counter()
        if game == "kuhn":
            store = train_kuhn_cfr(iteration_count, variant=variant)
            row = ComparisonRow(
                game=game,
                variant=variant.value,
                iterations=iteration_count,
                avg_value_p0=expected_game_value_for_average_strategy(store),
                root_bet_j=average_strategy_root_bet_probability(store, KuhnCard.JACK),
                root_bet_q=average_strategy_root_bet_probability(store, KuhnCard.QUEEN),
                root_bet_k=average_strategy_root_bet_probability(store, KuhnCard.KING),
                elapsed_seconds=perf_counter() - started_at,
            )
        else:
            store = train_leduc_cfr(iteration_count, variant=variant)
            row = ComparisonRow(
                game=game,
                variant=variant.value,
                iterations=iteration_count,
                avg_value_p0=expected_game_value_for_average_strategy_leduc(store),
                root_bet_j=average_strategy_root_bet_probability_leduc(
                    store,
                    LeducRank.JACK,
                ),
                root_bet_q=average_strategy_root_bet_probability_leduc(
                    store,
                    LeducRank.QUEEN,
                ),
                root_bet_k=average_strategy_root_bet_probability_leduc(
                    store,
                    LeducRank.KING,
                ),
                elapsed_seconds=perf_counter() - started_at,
            )
        rows.append(row)

    _write_rows(output_path, rows)
    if progress_callback is not None:
        progress_callback(len(tasks), len(tasks), f"saved:{output_path}")
    return rows


def _write_rows(output_path: Path, rows: Sequence[ComparisonRow]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "game",
                "variant",
                "iterations",
                "avg_value_p0",
                "root_bet_J",
                "root_bet_Q",
                "root_bet_K",
                "elapsed_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.game,
                    row.variant,
                    row.iterations,
                    f"{row.avg_value_p0:.12f}",
                    f"{row.root_bet_j:.12f}",
                    f"{row.root_bet_q:.12f}",
                    f"{row.root_bet_k:.12f}",
                    f"{row.elapsed_seconds:.6f}",
                ]
            )
