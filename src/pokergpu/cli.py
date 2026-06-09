import logging
import sys
from pathlib import Path

from .app import create_app
from .benchmarks import run_benchmark
from .cfr import (
    CFRVariant,
    KuhnCard,
    LeducRank,
    average_strategy_root_bet_probability,
    average_strategy_root_bet_probability_leduc,
    expected_game_value_for_average_strategy,
    expected_game_value_for_average_strategy_leduc,
    run_toy_game_comparison,
    train_kuhn_cfr,
    train_leduc_cfr,
)


def main() -> int:
    settings = create_app()
    logger = logging.getLogger(__name__)
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        result = run_benchmark("noop", lambda: None)
        print(
            "benchmark="
            f"{result.name} iterations={result.iterations} "
            f"seconds={result.total_seconds:.6f} "
            f"per_iter={result.seconds_per_iteration:.9f}"
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "compare-toy":
        output_path, kuhn_iterations, leduc_iterations, variants = _parse_compare_args(
            sys.argv[2:],
            default_output=settings.artifact_dir / "toy_game_comparison.csv",
        )
        run_toy_game_comparison(
            output_path=output_path,
            kuhn_iterations=kuhn_iterations,
            leduc_iterations=leduc_iterations,
            variants=variants,
            progress_callback=_print_progress,
        )
        print(f"output={output_path}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "kuhn":
        iterations, variant = _parse_solver_args(sys.argv[2:], default_iterations=2000)
        store = train_kuhn_cfr(iterations, variant=variant)
        print(f"variant={variant.value}")
        print(f"iterations={iterations}")
        print(
            "avg_value_p0="
            f"{expected_game_value_for_average_strategy(store):.12f}"
        )
        print(
            "root_bet_J="
            f"{average_strategy_root_bet_probability(store, KuhnCard.JACK):.12f}"
        )
        print(
            "root_bet_Q="
            f"{average_strategy_root_bet_probability(store, KuhnCard.QUEEN):.12f}"
        )
        print(
            "root_bet_K="
            f"{average_strategy_root_bet_probability(store, KuhnCard.KING):.12f}"
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "leduc":
        iterations, variant = _parse_solver_args(sys.argv[2:], default_iterations=800)
        store = train_leduc_cfr(iterations, variant=variant)
        jack_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.JACK)
        queen_bet = average_strategy_root_bet_probability_leduc(
            store,
            LeducRank.QUEEN,
        )
        king_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.KING)
        print(f"variant={variant.value}")
        print(f"iterations={iterations}")
        print(
            "avg_value_p0="
            f"{expected_game_value_for_average_strategy_leduc(store):.12f}"
        )
        print("root_bet_J=" f"{jack_bet:.12f}")
        print("root_bet_Q=" f"{queen_bet:.12f}")
        print("root_bet_K=" f"{king_bet:.12f}")
        return 0
    logger.info("PokerGPU initialized")
    print(f"PokerGPU ready on device={settings.device}")
    return 0


def _parse_solver_args(
    args: list[str],
    default_iterations: int,
) -> tuple[int, CFRVariant]:
    iterations = default_iterations
    variant = CFRVariant.VANILLA
    index = 0
    if index < len(args) and not args[index].startswith("--"):
        iterations = int(args[index])
        index += 1
    while index < len(args):
        if args[index] != "--variant" or index + 1 >= len(args):
            raise ValueError(f"invalid solver arguments: {args!r}")
        variant = CFRVariant(args[index + 1])
        index += 2
    return iterations, variant


def _parse_compare_args(
    args: list[str],
    default_output: Path,
) -> tuple[Path, tuple[int, ...], tuple[int, ...], tuple[CFRVariant, ...]]:
    output_path = default_output
    kuhn_iterations: tuple[int, ...] = (100, 500, 1000, 2000)
    leduc_iterations: tuple[int, ...] = (100, 300, 800)
    variants: tuple[CFRVariant, ...] = (
        CFRVariant.VANILLA,
        CFRVariant.CFR_PLUS,
        CFRVariant.DCFR,
    )
    index = 0
    while index < len(args):
        option = args[index]
        if option == "--output" and index + 1 < len(args):
            output_path = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--kuhn" and index + 1 < len(args):
            kuhn_iterations = _parse_iteration_list(args[index + 1])
            index += 2
            continue
        if option == "--leduc" and index + 1 < len(args):
            leduc_iterations = _parse_iteration_list(args[index + 1])
            index += 2
            continue
        if option == "--variants" and index + 1 < len(args):
            variants = tuple(CFRVariant(value) for value in args[index + 1].split(","))
            index += 2
            continue
        raise ValueError(f"invalid compare arguments: {args!r}")
    return output_path, kuhn_iterations, leduc_iterations, variants


def _parse_iteration_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in value.split(",") if item)
    if not items:
        raise ValueError("iteration list must not be empty")
    return items


def _print_progress(current: int, total: int, label: str) -> None:
    width = 24
    filled = width if total == 0 else int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rprogress [{bar}] {current}/{total} {label}",
        end="" if current < total else "\n",
        flush=True,
    )
