from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.signatures import public_state_signature
from pokergpu.core.state import GameState, HandPhase, PlayerState


def test_public_state_signature_excludes_private_cards() -> None:
    state = GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(
                player=PlayerIndex(0),
                hole_cards=None,
            ),
            PlayerState(
                player=PlayerIndex(1),
                hole_cards=None,
                folded=True,
            ),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(900)),
                PlayerStack(player=PlayerIndex(1), stack=chips(800)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(100)),
                PlayerBet(player=PlayerIndex(1), committed=chips(100), folded=True),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        phase=HandPhase.IN_PROGRESS,
        dealer=PlayerIndex(1),
    )

    signature = public_state_signature(state)

    assert "AhKdTc" in signature
    assert "folded=1" in signature
    assert "stack=900" in signature
    assert "committed=100" in signature
    assert "Qs" not in signature
    assert "Jh" not in signature
