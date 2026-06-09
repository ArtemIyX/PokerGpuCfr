import pytest

from pokergpu.core.cards import cards_from_str
from pokergpu.eval.treys_evaluator import (
    TreysHandEvaluator,
    evaluate_five_card_hand,
    evaluate_five_card_hands,
    evaluate_seven_card_hand,
    evaluate_seven_card_hands,
)


@pytest.mark.parametrize(
    ("cards", "expected_class_name"),
    [
        ("AhKhQhJhTh", "Royal Flush"),
        ("9h8h7h6h5h", "Straight Flush"),
        ("AhAdAcAs2d", "Four of a Kind"),
        ("AhAdAcKdKs", "Full House"),
        ("AhJh8h5h2h", "Flush"),
        ("9h8d7s6c5h", "Straight"),
        ("AhAdAc7c4s", "Three of a Kind"),
        ("AhAd7c7s4s", "Two Pair"),
        ("AhAd7c4s2d", "Pair"),
        ("AhKd9c7s3d", "High Card"),
    ],
)
def test_five_card_evaluator_known_class_names(
    cards: str,
    expected_class_name: str,
) -> None:
    result = evaluate_five_card_hand(cards_from_str(cards))

    assert result.class_name == expected_class_name


def test_five_card_evaluator_orders_hands_correctly() -> None:
    straight_flush = evaluate_five_card_hand(cards_from_str("AhKhQhJhTh"))
    pair = evaluate_five_card_hand(cards_from_str("AhAd7c4s2d"))

    assert straight_flush.score < pair.score
    assert straight_flush.class_name == "Royal Flush"


def test_seven_card_evaluator_orders_hands_correctly() -> None:
    trips = evaluate_seven_card_hand(cards_from_str("AhAdAc7c4s2d3h"))
    two_pair = evaluate_seven_card_hand(cards_from_str("AhAd7c7s4s2d3h"))

    assert trips.score < two_pair.score


def test_five_card_evaluator_known_ordering() -> None:
    ordered_scores = [
        evaluate_five_card_hand(cards_from_str(cards)).score
        for cards in (
            "AhKhQhJhTh",
            "9h8h7h6h5h",
            "AhAdAcAs2d",
            "AhAdAcKdKs",
            "AhJh8h5h2h",
            "9h8d7s6c5h",
            "AhAdAc7c4s",
            "AhAd7c7s4s",
            "AhAd7c4s2d",
            "AhKd9c7s3d",
        )
    ]

    assert ordered_scores == sorted(ordered_scores)


def test_seven_card_evaluator_selects_best_five_card_hand() -> None:
    result = evaluate_seven_card_hand(cards_from_str("AhKhQhJhTh2d3c"))

    assert result.class_name == "Royal Flush"


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


def test_batch_five_card_evaluation_returns_tuple() -> None:
    results = evaluate_five_card_hands(
        (
            cards_from_str("AhKhQhJhTh"),
            cards_from_str("AhAd7c4s2d"),
        )
    )

    assert len(results) == 2
    assert results[0].score < results[1].score


def test_batch_seven_card_evaluation_returns_tuple() -> None:
    results = evaluate_seven_card_hands(
        (
            cards_from_str("AhAdAc7c4s2d3h"),
            cards_from_str("AhAd7c7s4s2d3h"),
        )
    )

    assert len(results) == 2
    assert results[0].score < results[1].score
