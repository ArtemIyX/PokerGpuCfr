import logging
import sys

from .app import create_app
from .benchmarks import run_benchmark
from .cfr import (
    KuhnCard,
    LeducRank,
    average_strategy_root_bet_probability,
    average_strategy_root_bet_probability_leduc,
    expected_game_value_for_average_strategy,
    expected_game_value_for_average_strategy_leduc,
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
    if len(sys.argv) > 1 and sys.argv[1] == "kuhn":
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        store = train_kuhn_cfr(iterations)
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
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 800
        store = train_leduc_cfr(iterations)
        jack_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.JACK)
        queen_bet = average_strategy_root_bet_probability_leduc(
            store,
            LeducRank.QUEEN,
        )
        king_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.KING)
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
