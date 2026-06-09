import math

from pokergpu.cfr import (
    LeducAction,
    LeducCard,
    LeducRank,
    LeducState,
    average_strategy_root_bet_probability_leduc,
    expected_action_utilities_leduc,
    expected_game_value_for_average_strategy_leduc,
    leduc_infoset_layout,
    leduc_infosets,
    new_leduc_infoset_store,
    train_leduc_cfr,
)


def test_leduc_infoset_layout_is_non_empty() -> None:
    layout = leduc_infoset_layout()

    assert layout.infoset_count > 0
    assert layout.total_actions == layout.infoset_count * 2


def test_leduc_root_state_has_check_and_bet() -> None:
    state = LeducState(
        cards=(
            LeducCard(LeducRank.JACK, 0),
            LeducCard(LeducRank.KING, 0),
        )
    )

    assert state.legal_actions() == (LeducAction.CHECK, LeducAction.BET)
    assert state.player_to_act == 0


def test_leduc_preflop_check_check_reveals_public_card() -> None:
    state = LeducState(
        cards=(
            LeducCard(LeducRank.JACK, 0),
            LeducCard(LeducRank.KING, 0),
        )
    )

    state = state.apply_action(LeducAction.CHECK)
    state = state.apply_action(LeducAction.CHECK)

    assert state.needs_public_chance


def test_leduc_showdown_pair_beats_high_card() -> None:
    state = LeducState(
        cards=(
            LeducCard(LeducRank.JACK, 0),
            LeducCard(LeducRank.KING, 0),
        ),
        public_card=LeducCard(LeducRank.JACK, 1),
        round_index=1,
    )
    state = state.apply_action(LeducAction.CHECK)
    state = state.apply_action(LeducAction.CHECK)

    assert state.is_terminal
    assert math.isclose(state.payoff(0), 1.0)
    assert math.isclose(state.payoff(1), -1.0)


def test_leduc_expected_action_utilities_match_layout() -> None:
    store = new_leduc_infoset_store()

    utilities = expected_action_utilities_leduc(store, updating_player=0)

    assert len(utilities) == leduc_infoset_layout().infoset_count
    assert all(values.shape == (2,) for values in utilities)


def test_leduc_root_infosets_exist() -> None:
    root_infosets = [
        infoset
        for infoset in leduc_infosets()
        if infoset.player == 0 and infoset.round_index == 0 and infoset.history == ()
    ]

    assert len(root_infosets) == 3


def test_leduc_cfr_average_strategy_value_stabilizes() -> None:
    smaller = train_leduc_cfr(200)
    larger = train_leduc_cfr(800)

    value_small = expected_game_value_for_average_strategy_leduc(smaller)
    value_large = expected_game_value_for_average_strategy_leduc(larger)

    assert math.isfinite(value_small)
    assert math.isfinite(value_large)
    assert abs(value_large - value_small) < 0.5


def test_leduc_cfr_root_betting_sanity_orders_ranks() -> None:
    store = train_leduc_cfr(800)

    jack_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.JACK)
    queen_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.QUEEN)
    king_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.KING)

    assert 0.0 <= queen_bet <= 1.0
    assert 0.0 <= jack_bet <= 1.0
    assert 0.0 <= king_bet <= 1.0
    assert king_bet >= queen_bet
