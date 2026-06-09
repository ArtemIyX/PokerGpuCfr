from random import Random

import pytest

from pokergpu.cards import (
    Card,
    Rank,
    Suit,
    card_from_str,
    cards_from_str,
    format_cards,
    make_deck,
    shuffled_deck,
)


def test_card_from_str_parses_rank_and_suit() -> None:
    card = Card.from_str("Ah")

    assert card.rank is Rank.ACE
    assert card.suit is Suit.HEARTS
    assert str(card) == "Ah"


def test_card_from_str_rejects_invalid_literal() -> None:
    with pytest.raises(ValueError):
        Card.from_str("A")


def test_card_helper_round_trip() -> None:
    cards = cards_from_str("AhKdTc")

    assert cards == (
        card_from_str("Ah"),
        card_from_str("Kd"),
        card_from_str("Tc"),
    )
    assert format_cards(list(cards)) == "AhKdTc"


def test_cards_from_str_rejects_odd_length() -> None:
    with pytest.raises(ValueError):
        cards_from_str("AhK")


def test_make_deck_has_52_unique_cards() -> None:
    deck = make_deck()

    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_shuffled_deck_is_deterministic_with_rng() -> None:
    first = shuffled_deck(Random(7))
    second = shuffled_deck(Random(7))

    assert first == second
    assert sorted(first) == sorted(make_deck())
