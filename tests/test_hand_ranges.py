import math

import pytest

from pokergpu.abstraction import (
    PrivateHand,
    RangeVector,
    all_private_hands,
    private_hand_count,
    private_hand_from_index,
    private_hand_index,
    private_hand_mask,
)
from pokergpu.core.cards import card_from_str


def test_private_hand_count_matches_holdem_combo_count() -> None:
    assert private_hand_count() == 1326


def test_private_hand_index_round_trips() -> None:
    ace_hearts = card_from_str("Ah")
    king_diamonds = card_from_str("Kd")
    index = private_hand_index(ace_hearts, king_diamonds)

    assert private_hand_from_index(index) == PrivateHand.from_cards(
        ace_hearts,
        king_diamonds,
    )


def test_private_hand_index_is_order_independent() -> None:
    ace_hearts = card_from_str("Ah")
    king_diamonds = card_from_str("Kd")

    assert private_hand_index(ace_hearts, king_diamonds) == private_hand_index(
        king_diamonds,
        ace_hearts,
    )


def test_private_hand_mask_excludes_dead_cards() -> None:
    ace_hearts = card_from_str("Ah")
    king_diamonds = card_from_str("Kd")
    mask = private_hand_mask([ace_hearts])

    assert not bool(mask[int(private_hand_index(ace_hearts, king_diamonds))])
    assert bool(mask[int(private_hand_index(card_from_str("Qc"), card_from_str("Js")))])


def test_private_hand_mask_rejects_duplicate_dead_cards() -> None:
    ace_hearts = card_from_str("Ah")

    with pytest.raises(ValueError):
        private_hand_mask([ace_hearts, ace_hearts])


def test_uniform_range_sums_to_one() -> None:
    hand_range = RangeVector.uniform()

    assert math.isclose(hand_range.total_weight(), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_normalized_range_scales_weights() -> None:
    values = [0.0] * private_hand_count()
    values[0] = 2.0
    values[1] = 6.0

    hand_range = RangeVector.from_values(values).normalized()

    assert math.isclose(hand_range.total_weight(), 1.0)
    assert math.isclose(float(hand_range.values[0]), 0.25)
    assert math.isclose(float(hand_range.values[1]), 0.75)


def test_normalized_rejects_zero_weight_range() -> None:
    with pytest.raises(ValueError):
        RangeVector.zeros().normalized()


def test_masked_range_zeros_blocked_combos() -> None:
    ace_hearts = card_from_str("Ah")
    king_diamonds = card_from_str("Kd")
    queen_clubs = card_from_str("Qc")
    jack_spades = card_from_str("Js")
    values = [0.0] * private_hand_count()
    blocked_index = int(private_hand_index(ace_hearts, king_diamonds))
    live_index = int(private_hand_index(queen_clubs, jack_spades))
    values[blocked_index] = 3.0
    values[live_index] = 5.0

    masked = RangeVector.from_values(values).masked([ace_hearts])

    assert math.isclose(float(masked.values[blocked_index]), 0.0)
    assert math.isclose(float(masked.values[live_index]), 5.0)


def test_all_private_hands_are_unique() -> None:
    hands = all_private_hands()

    assert len(hands) == len(set(hands))
