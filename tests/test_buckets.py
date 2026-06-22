import math

import pytest

from pokergpu.abstraction import (
    PrivateHand,
    PreflopClassBucketer,
    RangeVector,
    StrengthTierBucketer,
    private_hand_index,
)
from pokergpu.core.board import board_from_str
from pokergpu.core.cards import card_from_str


def test_strength_tier_bucketer_rejects_preflop_board() -> None:
    bucketer = StrengthTierBucketer()

    with pytest.raises(ValueError):
        bucketer.bucket_mask(board_from_str(""))


def test_strength_tier_bucketer_rejects_board_overlap() -> None:
    bucketer = StrengthTierBucketer()
    board = board_from_str("AhKhQh")

    with pytest.raises(ValueError):
        bucketer.bucket_for_hand(
            hand=PrivateHand.from_cards(card_from_str("Ah"), card_from_str("2c")),
            board=board,
        )


def test_strength_tier_bucketer_distinguishes_strong_and_weak_showdown_hands() -> None:
    bucketer = StrengthTierBucketer()
    board = board_from_str("QhJhTh9c2d")

    strong_bucket = bucketer.bucket_for_hand(
        hand=PrivateHand.from_cards(card_from_str("Ah"), card_from_str("Kh")),
        board=board,
    )
    weak_bucket = bucketer.bucket_for_hand(
        hand=PrivateHand.from_cards(card_from_str("3c"), card_from_str("4d")),
        board=board,
    )

    assert int(strong_bucket) > int(weak_bucket)


def test_bucketed_range_accumulates_live_combo_weights() -> None:
    bucketer = StrengthTierBucketer()
    board = board_from_str("QhJhTh9c2d")
    values = [0.0] * 1326
    strong_index = int(private_hand_index(card_from_str("Ah"), card_from_str("Kh")))
    weak_index = int(private_hand_index(card_from_str("3c"), card_from_str("4d")))
    blocked_index = int(private_hand_index(card_from_str("Qh"), card_from_str("2c")))
    values[strong_index] = 0.4
    values[weak_index] = 0.6
    values[blocked_index] = 0.9

    bucketed = bucketer.bucketed_range(RangeVector.from_values(values), board)
    assignments = bucketer.bucket_mask(board)

    assert math.isclose(float(bucketed.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        float(bucketed[int(assignments[strong_index])]),
        0.4,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(bucketed[int(assignments[weak_index])]),
        0.6,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_preflop_class_bucketer_is_deterministic_and_fixed_width() -> None:
    bucketer = PreflopClassBucketer()

    pocket_aces = PrivateHand.from_cards(card_from_str("Ah"), card_from_str("Ad"))
    ace_king_suited = PrivateHand.from_cards(card_from_str("Ah"), card_from_str("Kh"))
    seven_two_offsuit = PrivateHand.from_cards(card_from_str("7c"), card_from_str("2d"))

    assert bucketer.bucket_count == 8
    assert int(bucketer.bucket_for_hand(pocket_aces)) > int(bucketer.bucket_for_hand(ace_king_suited))
    assert int(bucketer.bucket_for_hand(ace_king_suited)) > int(bucketer.bucket_for_hand(seven_two_offsuit))


def test_preflop_class_bucketer_bucketed_range_is_dense() -> None:
    bucketer = PreflopClassBucketer()
    values = [0.0] * 1326
    strong_index = int(private_hand_index(card_from_str("Ah"), card_from_str("Ad")))
    weak_index = int(private_hand_index(card_from_str("7c"), card_from_str("2d")))
    values[strong_index] = 0.4
    values[weak_index] = 0.6

    bucketed = bucketer.bucketed_range(RangeVector.from_values(values))

    assert bucketed.shape == (8,)
    assert math.isclose(float(bucketed.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert bucketed[int(bucketer.bucket_for_hand(PrivateHand.from_cards(card_from_str("Ah"), card_from_str("Ad"))))] > 0.0


def test_preflop_class_bucketer_marks_blocked_hands() -> None:
    bucketer = PreflopClassBucketer()
    blocked = bucketer.bucket_mask([card_from_str("Ah")])

    assert blocked.shape == (1326,)
    assert blocked[int(private_hand_index(card_from_str("Ah"), card_from_str("Kd")))] == -1
    assert blocked[int(private_hand_index(card_from_str("Qc"), card_from_str("Js")))] >= 0
