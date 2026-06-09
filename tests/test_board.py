import pytest

from pokergpu.core.board import Board, Street, board_from_str


def test_empty_board_is_preflop() -> None:
    board = Board(cards=())

    assert board.street is Street.PREFLOP
    assert board.is_preflop
    assert str(board) == ""


def test_flop_board_parses() -> None:
    board = Board.from_str("AhKdTc")

    assert board.street is Street.FLOP
    assert board.is_flop
    assert str(board) == "AhKdTc"


def test_turn_board_parses() -> None:
    board = board_from_str("AhKdTc2s")

    assert board.street is Street.TURN
    assert board.is_turn


def test_river_board_parses() -> None:
    board = board_from_str("AhKdTc2s3d")

    assert board.street is Street.RIVER
    assert board.is_river


def test_board_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        Board.from_str("AhKd")


def test_board_rejects_duplicate_cards() -> None:
    with pytest.raises(ValueError):
        Board.from_str("AhAhTc")
