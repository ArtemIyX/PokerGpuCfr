import math

import numpy as np

from pokergpu.cfr import (
    CFRVariant,
    KuhnAction,
    KuhnCard,
    KuhnState,
    average_strategy_root_bet_probability,
    expected_action_utilities,
    expected_game_value_for_average_strategy,
    kuhn_infoset_layout,
    kuhn_infosets,
    new_kuhn_infoset_store,
    train_kuhn_cfr,
)


def test_kuhn_infoset_layout_has_expected_size() -> None:
    layout = kuhn_infoset_layout()

    assert layout.infoset_count == 12
    assert layout.total_actions == 24


def test_kuhn_root_state_has_check_and_bet() -> None:
    state = KuhnState(cards=(KuhnCard.JACK, KuhnCard.KING))

    assert state.legal_actions() == (KuhnAction.CHECK, KuhnAction.BET)
    assert state.player_to_act == 0


def test_kuhn_bet_fold_payoff() -> None:
    state = KuhnState(
        cards=(KuhnCard.JACK, KuhnCard.KING),
        history=(KuhnAction.BET, KuhnAction.FOLD),
    )

    assert state.is_terminal
    assert math.isclose(state.payoff(0), 1.0)
    assert math.isclose(state.payoff(1), -1.0)


def test_kuhn_showdown_payoff_with_single_bet_round() -> None:
    state = KuhnState(
        cards=(KuhnCard.KING, KuhnCard.JACK),
        history=(KuhnAction.CHECK, KuhnAction.CHECK),
    )

    assert state.is_terminal
    assert math.isclose(state.payoff(0), 1.0)
    assert math.isclose(state.payoff(1), -1.0)


def test_kuhn_called_bet_has_two_chip_showdown_payoff() -> None:
    state = KuhnState(
        cards=(KuhnCard.JACK, KuhnCard.KING),
        history=(KuhnAction.BET, KuhnAction.CALL),
    )

    assert state.is_terminal
    assert math.isclose(state.payoff(0), -2.0)
    assert math.isclose(state.payoff(1), 2.0)


def test_kuhn_expected_action_utilities_match_layout() -> None:
    store = new_kuhn_infoset_store()

    utilities = expected_action_utilities(store, updating_player=0)

    assert len(utilities) == kuhn_infoset_layout().infoset_count
    assert all(values.shape == (2,) for values in utilities)


def test_kuhn_expected_action_utilities_are_non_zero_for_root_infosets() -> None:
    store = new_kuhn_infoset_store()

    utilities = expected_action_utilities(store, updating_player=0)
    infosets = kuhn_infosets()
    root_indices = [
        index
        for index, infoset in enumerate(infosets)
        if infoset.player == 0 and infoset.history == ()
    ]

    assert len(root_indices) == 3
    assert any(
        not np.allclose(utilities[index], np.zeros(2, dtype=np.float32))
        for index in root_indices
    )


def test_kuhn_cfr_average_strategy_value_approaches_known_game_value() -> None:
    store = train_kuhn_cfr(2000)

    value = expected_game_value_for_average_strategy(store)

    assert math.isclose(value, -1.0 / 18.0, rel_tol=0.0, abs_tol=0.08)


def test_kuhn_cfr_root_betting_profile_matches_sanity_pattern() -> None:
    store = train_kuhn_cfr(2000)

    jack_bet = average_strategy_root_bet_probability(store, KuhnCard.JACK)
    queen_bet = average_strategy_root_bet_probability(store, KuhnCard.QUEEN)
    king_bet = average_strategy_root_bet_probability(store, KuhnCard.KING)

    assert 0.0 <= queen_bet < jack_bet < king_bet <= 1.0


def test_kuhn_cfr_plus_average_strategy_value_approaches_known_game_value() -> None:
    store = train_kuhn_cfr(2000, variant=CFRVariant.CFR_PLUS)

    value = expected_game_value_for_average_strategy(store)

    assert math.isclose(value, -1.0 / 18.0, rel_tol=0.0, abs_tol=0.08)
