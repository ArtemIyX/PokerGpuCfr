from typing import Any

import pytest

from pokergpu.core.cards import cards_from_str
from pokergpu.eval.treys_evaluator import TreysHandEvaluator

pytestmark = pytest.mark.benchmark_suite

FIVE_CARD_HAND = cards_from_str("AhKhQhJhTh")
SEVEN_CARD_HAND = cards_from_str("AhKhQhJhTh2d3c")
FIVE_CARD_BATCH = (
    cards_from_str("AhKhQhJhTh"),
    cards_from_str("AhAd7c4s2d"),
    cards_from_str("9h8h7h6h5h"),
    cards_from_str("AhAdAcAs2d"),
    cards_from_str("AhKd9c7s3d"),
) * 200
SEVEN_CARD_BATCH = (
    cards_from_str("AhKhQhJhTh2d3c"),
    cards_from_str("AhAdAc7c4s2d3h"),
    cards_from_str("AhAd7c7s4s2d3h"),
    cards_from_str("9h8h7h6h5h2d3c"),
    cards_from_str("AhKd9c7s3d2c4h"),
) * 200


def test_benchmark_single_five_card_evaluation(benchmark: Any) -> None:
    evaluator = TreysHandEvaluator()

    result = benchmark(evaluator.evaluate_five_card_hand, FIVE_CARD_HAND)

    assert result.score > 0


def test_benchmark_single_seven_card_evaluation(benchmark: Any) -> None:
    evaluator = TreysHandEvaluator()

    result = benchmark(evaluator.evaluate_seven_card_hand, SEVEN_CARD_HAND)

    assert result.score > 0


def test_benchmark_batch_five_card_evaluation(benchmark: Any) -> None:
    evaluator = TreysHandEvaluator()

    result = benchmark(evaluator.evaluate_five_card_hands, FIVE_CARD_BATCH)

    assert len(result) == len(FIVE_CARD_BATCH)


def test_benchmark_batch_seven_card_evaluation(benchmark: Any) -> None:
    evaluator = TreysHandEvaluator()

    result = benchmark(evaluator.evaluate_seven_card_hands, SEVEN_CARD_BATCH)

    assert len(result) == len(SEVEN_CARD_BATCH)
