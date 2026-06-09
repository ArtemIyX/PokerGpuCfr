import pytest

from pokergpu.core.cards import cards_from_str
from pokergpu.eval.treys_evaluator import (
    TreysHandEvaluator,
    evaluate_five_card_hand,
    evaluate_seven_card_hand,
)


def test_five_card_evaluator_orders_hands_correctly() -> None:
    straight_flush = evaluate_five_card_hand(cards_from_str("AhKhQhJhTh"))
    pair = evaluate_five_card_hand(cards_from_str("AhAd7c4s2d"))

    assert straight_flush.score < pair.score
    assert straight_flush.class_name == "Royal Flush"


def test_seven_card_evaluator_orders_hands_correctly() -> None:
    trips = evaluate_seven_card_hand(cards_from_str("AhAdAc7c4s2d3h"))
    two_pair = evaluate_seven_card_hand(cards_from_str("AhAd7c7s4s2d3h"))

    assert trips.score < two_pair.score


def test_evaluator_rejects_wrong_card_count() -> None:
    evaluator = TreysHandEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate_five_card_hand(cards_from_str("AhKhQhJh"))

    with pytest.raises(ValueError):
        evaluator.evaluate_seven_card_hand(cards_from_str("AhKhQhJhTh9d"))


def test_evaluator_rejects_duplicate_cards() -> None:
    evaluator = TreysHandEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate_five_card_hand(cards_from_str("AhAhQhJhTh"))
