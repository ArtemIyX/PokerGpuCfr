import math

import pytest

from pokergpu.abstraction import (
    PlayerRangeVectors,
    PrivateHand,
    RangeVector,
    all_private_hands,
    apply_board_dead_cards,
    apply_dead_cards,
    canonicalize_private_hand_for_board,
    masked_range_vector,
    normalize_range_vector,
    propagate_player_ranges,
    private_hand_count,
    private_hand_from_index,
    private_hand_index,
    private_hand_mask,
)
from pokergpu.core.board import Board
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


def test_masked_range_normalizes_after_board_cards() -> None:
    values = [0.0] * private_hand_count()
    live_index = int(private_hand_index(card_from_str("Qc"), card_from_str("Js")))
    values[live_index] = 7.0

    masked = RangeVector.from_values(values).normalized_masked([card_from_str("Ah")])

    assert math.isclose(masked.total_weight(), 1.0, abs_tol=1e-6)
    assert math.isclose(float(masked.values[live_index]), 1.0, abs_tol=1e-6)


def test_player_range_vectors_apply_dead_cards_to_each_player() -> None:
    values0 = [0.0] * private_hand_count()
    values1 = [0.0] * private_hand_count()
    index0 = int(private_hand_index(card_from_str("Ah"), card_from_str("Kd")))
    index1 = int(private_hand_index(card_from_str("Qc"), card_from_str("Js")))
    values0[index0] = 2.0
    values1[index1] = 3.0
    player_ranges = PlayerRangeVectors.from_values(
        (
            RangeVector.from_values(values0),
            RangeVector.from_values(values1),
        )
    )

    masked = player_ranges.masked([card_from_str("Ah")])

    assert math.isclose(float(masked.values[0].values[index0]), 0.0)
    assert math.isclose(float(masked.values[1].values[index1]), 3.0)


def test_player_range_vectors_normalize_after_masking() -> None:
    values = [0.0] * private_hand_count()
    live_index = int(private_hand_index(card_from_str("Qc"), card_from_str("Js")))
    values[live_index] = 5.0
    player_ranges = PlayerRangeVectors.from_values(
        (RangeVector.from_values(values), RangeVector.from_values(values))
    )

    normalized = player_ranges.normalized_masked([card_from_str("Ah")])

    assert all(
        math.isclose(weight, 1.0, abs_tol=1e-6)
        for weight in normalized.total_weights()
    )


def test_normalize_range_vector_helper_matches_method() -> None:
    values = [0.0] * private_hand_count()
    values[0] = 4.0
    helper = normalize_range_vector(RangeVector.from_values(values))

    assert math.isclose(helper.total_weight(), 1.0, abs_tol=1e-6)


def test_masked_range_vector_helper_matches_method() -> None:
    values = [0.0] * private_hand_count()
    index = int(private_hand_index(card_from_str("Qc"), card_from_str("Js")))
    values[index] = 1.0
    helper = masked_range_vector(
        RangeVector.from_values(values),
        [card_from_str("Ah")],
    )

    assert math.isclose(float(helper.values[index]), 1.0, abs_tol=1e-6)


def test_apply_dead_cards_masks_all_ranges() -> None:
    values = [0.0] * private_hand_count()
    index = int(private_hand_index(card_from_str("Ah"), card_from_str("Kd")))
    values[index] = 1.0
    masked = apply_dead_cards(
        (RangeVector.from_values(values), RangeVector.from_values(values)),
        [card_from_str("Ah")],
    )

    assert math.isclose(float(masked[0].values[index]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(masked[1].values[index]), 0.0, abs_tol=1e-6)


def test_board_masking_keeps_total_mass_valid_after_renormalization() -> None:
    board = Board.from_str("AhKdQc")
    values = [0.0] * private_hand_count()
    live_index = int(private_hand_index(card_from_str("Js"), card_from_str("Ts")))
    blocked_index = int(private_hand_index(card_from_str("Ah"), card_from_str("Qs")))
    values[live_index] = 9.0
    values[blocked_index] = 4.0

    normalized = RangeVector.from_values(values).normalized_masked(board.cards)

    assert math.isclose(normalized.total_weight(), 1.0, abs_tol=1e-6)
    assert math.isclose(float(normalized.values[blocked_index]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(normalized.values[live_index]), 1.0, abs_tol=1e-6)


def test_apply_board_dead_cards_normalizes_after_filtering() -> None:
    board = Board.from_str("AhKdQc")
    values = [0.0] * private_hand_count()
    blocked_index = int(private_hand_index(card_from_str("Ah"), card_from_str("Js")))
    live_index = int(private_hand_index(card_from_str("9c"), card_from_str("8d")))
    values[blocked_index] = 2.0
    values[live_index] = 6.0

    filtered = apply_board_dead_cards(RangeVector.from_values(values), board)

    assert math.isclose(filtered.total_weight(), 1.0, abs_tol=1e-6)
    assert math.isclose(float(filtered.values[blocked_index]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(filtered.values[live_index]), 1.0, abs_tol=1e-6)


def test_player_range_vectors_mask_and_normalize_each_player() -> None:
    board = Board.from_str("AhKdQc")
    values0 = [0.0] * private_hand_count()
    values1 = [0.0] * private_hand_count()
    live0 = int(private_hand_index(card_from_str("Js"), card_from_str("Ts")))
    live1 = int(private_hand_index(card_from_str("9c"), card_from_str("8d")))
    dead = int(private_hand_index(card_from_str("Ah"), card_from_str("Qs")))
    values0[live0] = 5.0
    values0[dead] = 2.0
    values1[live1] = 7.0
    values1[dead] = 3.0

    ranges = PlayerRangeVectors.from_values(
        (
            RangeVector.from_values(values0),
            RangeVector.from_values(values1),
            RangeVector.uniform(),
        )
    )

    masked = ranges.normalized_masked(board.cards)

    assert len(masked.values) == 3
    assert math.isclose(masked.values[0].total_weight(), 1.0, abs_tol=1e-6)
    assert math.isclose(masked.values[1].total_weight(), 1.0, abs_tol=1e-6)
    assert math.isclose(masked.values[2].total_weight(), 1.0, abs_tol=1e-6)
    assert math.isclose(float(masked.values[0].values[dead]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(masked.values[1].values[dead]), 0.0, abs_tol=1e-6)


def test_three_way_range_propagation_masks_dead_cards() -> None:
    board = Board.from_str("AhKdQc")
    values0 = [0.0] * private_hand_count()
    values1 = [0.0] * private_hand_count()
    values2 = [0.0] * private_hand_count()
    live = int(private_hand_index(card_from_str("Js"), card_from_str("Ts")))
    blocked = int(private_hand_index(card_from_str("Ah"), card_from_str("Qs")))
    values0[live] = 5.0
    values1[live] = 7.0
    values2[live] = 9.0
    values0[blocked] = 2.0
    values1[blocked] = 3.0
    values2[blocked] = 4.0
    ranges = PlayerRangeVectors.from_values(
        (
            RangeVector.from_values(values0),
            RangeVector.from_values(values1),
            RangeVector.from_values(values2),
        )
    )

    propagated = propagate_player_ranges(ranges, board.cards)

    assert len(propagated.values) == 3
    assert math.isclose(float(propagated.values[0].values[blocked]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(propagated.values[1].values[blocked]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(propagated.values[2].values[blocked]), 0.0, abs_tol=1e-6)


def test_three_way_range_propagation_renormalizes() -> None:
    board = Board.from_str("AhKdQc")
    values = [0.0] * private_hand_count()
    live = int(private_hand_index(card_from_str("Js"), card_from_str("Ts")))
    values[live] = 5.0
    ranges = PlayerRangeVectors.from_values(
        (
            RangeVector.from_values(values),
            RangeVector.from_values(values),
            RangeVector.from_values(values),
        )
    )

    propagated = propagate_player_ranges(ranges, board.cards)

    assert all(math.isclose(rv.total_weight(), 1.0, abs_tol=1e-6) for rv in propagated.values)


def test_canonicalize_private_hand_for_board_uses_board_suit_map() -> None:
    board = Board.from_str("AhKhQd")
    hand = PrivateHand.from_cards(card_from_str("As"), card_from_str("2c"))

    canonical = canonicalize_private_hand_for_board(hand, board)

    assert {str(canonical.first)[1], str(canonical.second)[1]} == {"s", "h"}


def test_all_private_hands_are_unique() -> None:
    hands = all_private_hands()

    assert len(hands) == len(set(hands))
