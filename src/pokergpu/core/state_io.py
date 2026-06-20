from __future__ import annotations

import json
from random import Random

from .betting import BettingRoundState
from .betting import BlindStructure
from .betting import PlayerBet
from .betting import PlayerIndex
from .betting import PlayerStack
from .betting import Pot
from .betting import chips
from .board import Board
from .cards import Card
from .cards import shuffled_deck
from .state import GameState
from .state import HandPhase
from .state import PlayerState


def encode_game_state(state: GameState) -> bytes:
    payload = {
        "board": str(state.board),
        "phase": state.phase.value,
        "dealer": int(state.dealer),
        "players": [
            {
                "player": int(player.player),
                "hole_cards": [str(card) for card in player.hole_cards] if player.hole_cards is not None else None,
                "folded": player.folded,
                "all_in": player.all_in,
            }
            for player in state.players
        ],
        "stacks": [
            {"player": int(stack.player), "stack": int(stack.stack)}
            for stack in state.betting_round.stacks
        ],
        "bets": [
            {
                "player": int(bet.player),
                "committed": int(bet.committed),
                "folded": bet.folded,
                "all_in": bet.all_in,
            }
            for bet in state.betting_round.bets
        ],
        "blinds": {
            "small_blind": int(state.betting_round.blinds.small_blind),
            "big_blind": int(state.betting_round.blinds.big_blind),
            "ante": int(state.betting_round.blinds.ante),
        },
        "pot": int(state.betting_round.pot.amount),
        "to_act": int(state.betting_round.to_act),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_game_state(encoded_state: bytes) -> GameState | None:
    try:
        payload = json.loads(encoded_state.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        board = Board.from_str(str(payload.get("board", "")))
        players = tuple(
            PlayerState(
                player=PlayerIndex(int(player_payload["player"])),
                hole_cards=_parse_hole_cards(player_payload.get("hole_cards")),
                folded=bool(player_payload.get("folded", False)),
                all_in=bool(player_payload.get("all_in", False)),
            )
            for player_payload in payload.get("players", [])
        )
        stacks = tuple(
            PlayerStack(
                player=PlayerIndex(int(stack_payload["player"])),
                stack=chips(int(stack_payload["stack"])),
            )
            for stack_payload in payload.get("stacks", [])
        )
        bets = tuple(
            PlayerBet(
                player=PlayerIndex(int(bet_payload["player"])),
                committed=chips(int(bet_payload.get("committed", 0))),
                folded=bool(bet_payload.get("folded", False)),
                all_in=bool(bet_payload.get("all_in", False)),
            )
            for bet_payload in payload.get("bets", [])
        )
        blinds_payload = payload.get("blinds", {})
        blinds = BlindStructure(
            small_blind=chips(int(blinds_payload.get("small_blind", 1))),
            big_blind=chips(int(blinds_payload.get("big_blind", 2))),
            ante=chips(int(blinds_payload.get("ante", 0))),
        )
        betting_round = BettingRoundState(
            pot=Pot(amount=chips(int(payload.get("pot", 0)))),
            stacks=stacks,
            bets=bets,
            blinds=blinds,
            to_act=PlayerIndex(int(payload.get("to_act", 0))),
        )
        return GameState(
            board=board,
            players=players,
            betting_round=betting_round,
            phase=HandPhase(str(payload.get("phase", HandPhase.IN_PROGRESS.value))),
            dealer=PlayerIndex(int(payload.get("dealer", 0))),
        )
    except (KeyError, TypeError, ValueError):
        return None


def make_random_game_state(*, rng: Random, player_count: int = 2) -> GameState:
    if player_count != 2:
        raise ValueError("random debug state currently supports exactly 2 players")
    deck = shuffled_deck(rng)
    board = Board(cards=tuple())
    players = [
        PlayerState(player=PlayerIndex(0), hole_cards=(deck[0], deck[1])),
        PlayerState(player=PlayerIndex(1), hole_cards=(deck[2], deck[3])),
    ]
    stacks = (
        PlayerStack(player=PlayerIndex(0), stack=chips(100)),
        PlayerStack(player=PlayerIndex(1), stack=chips(100)),
    )
    bets = (
        PlayerBet(player=PlayerIndex(0), committed=chips(1)),
        PlayerBet(player=PlayerIndex(1), committed=chips(2)),
    )
    betting_round = BettingRoundState(
        pot=Pot(amount=chips(3)),
        stacks=stacks,
        bets=bets,
        blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
        to_act=PlayerIndex(rng.randrange(player_count)),
    )
    return GameState(board=board, players=tuple(players), betting_round=betting_round)


def _parse_hole_cards(value: object) -> tuple[Card, Card] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    return (Card.from_str(str(value[0])), Card.from_str(str(value[1])))
