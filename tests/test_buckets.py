import math

import pytest

from pokergpu.abstraction import (
    PrivateHand,
    RangeVector,
    StrengthTierBucketer,
    board_bucket_signature,
    private_hand_index,
)
from pokergpu.core.board import board_from_str
from pokergpu.core.canonical import canonical_board_key, canonicalize_board
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


def test_suit_isomorphic_boards_share_canonical_key() -> None:
    board_a = board_from_str("AhKhQd")
    board_b = board_from_str("AcKcQd")

    assert canonical_board_key(board_a) == canonical_board_key(board_b)
    assert board_bucket_signature(board_a) == board_bucket_signature(board_b)
    assert (canonicalize_board(board_a).canonical_key
            == canonicalize_board(board_b).canonical_key)


def test_non_isomorphic_boards_do_not_share_canonical_key() -> None:
    board_a = board_from_str("AhKhQh")
    board_b = board_from_str("AhKhQd")

    assert canonical_board_key(board_a) != canonical_board_key(board_b)
