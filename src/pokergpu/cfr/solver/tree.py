from __future__ import annotations

from pokergpu.abstraction.actions import BaselineActionAbstraction
from pokergpu.abstraction.actions import make_compact_profile
from pokergpu.abstraction.actions import make_holdem_hu_profile
from pokergpu.core.board import Board
from pokergpu.core.betting import BettingRoundState
from pokergpu.core.betting import BlindStructure
from pokergpu.core.betting import PlayerBet
from pokergpu.core.betting import PlayerIndex
from pokergpu.core.betting import PlayerStack
from pokergpu.core.betting import Pot
from pokergpu.core.betting import Chips
from pokergpu.core.betting import chips
from pokergpu.core.cards import Card
from pokergpu.core.state import GameState
from pokergpu.core.state import HandPhase
from pokergpu.core.state import PlayerState
from pokergpu.cfr.solver.spec import GameStateMode
from pokergpu.cfr.solver.spec import GameStateSpec
from pokergpu.cfr.solver.spec import GameVariant
from pokergpu.cfr.solver.kuhn import make_kuhn_public_tree
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.tree.builder import ChanceOutcome
from pokergpu.tree.builder import TreeBuildConfig
from pokergpu.tree.builder import build_public_tree
from pokergpu.tree.public_tree import ChildLink
from pokergpu.tree.public_tree import InfosetId
from pokergpu.tree.public_tree import NodeId
from pokergpu.tree.public_tree import NodeType
from pokergpu.tree.public_tree import PublicTree


def make_toy_public_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, Chips(1), Chips(-1)),
    )


def make_toy_pipeline_tree() -> PublicTree:
    return PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.PLAYER1,
            NodeType.LEAF,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 3, 3),
        child_count=(2, 1, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(3)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), InfosetId(1), None, None),
        terminal_payoffs=(None, None, None, Chips(2)),
    )


def make_game_public_tree(
    game: GameVariant,
    *,
    compact: bool = False,
    depth_limit: int | None = None,
) -> PublicTree:
    if game is GameVariant.KUHN:
        return make_kuhn_public_tree()
    if game is GameVariant.LEDUC:
        return make_leduc_public_tree()
    if game is GameVariant.HOLDEM_HU:
        return make_holdem_hu_public_tree(compact=compact or (depth_limit is not None and depth_limit <= 1))
    supported = ", ".join(
        (
            GameVariant.KUHN.value,
            GameVariant.LEDUC.value,
            GameVariant.HOLDEM_HU.value,
        )
    )
    raise NotImplementedError(f"unsupported game variant {game.value!r}; supported variants: {supported}")


def make_holdem_hu_public_tree(*, compact: bool = False) -> PublicTree:
    state = _make_canonical_holdem_state()
    abstraction = BaselineActionAbstraction(profile=make_compact_profile() if compact else make_holdem_hu_profile())
    config = TreeBuildConfig(max_depth=1 if compact else 2, max_nodes=256 if compact else 512)
    return build_public_tree(
        state,
        abstraction=abstraction,
        config=config,
        advance_next_board=_next_holdem_board if not compact else None,
        expand_chance=_expand_holdem_root_chance if not compact else None,
    ).tree


def _make_canonical_holdem_state() -> GameState:
    return GameState(
        board=Board(cards=()),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(3)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1000)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1000)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(1)),
                PlayerBet(player=PlayerIndex(1), committed=chips(2)),
            ),
            blinds=BlindStructure(small_blind=chips(1), big_blind=chips(2)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )


def _next_holdem_board(state: GameState) -> Board | None:
    if state.board.is_preflop:
        return Board.from_str("AhKdTc")
    if state.board.is_flop:
        return Board.from_str("AhKdTc9s")
    if state.board.is_turn:
        return Board.from_str("AhKdTc9s2d")
    return None


def _expand_holdem_root_chance(state: GameState) -> tuple[ChanceOutcome, ...] | None:
    if state.board.cards:
        return None
    if state.phase is not HandPhase.IN_PROGRESS:
        return None
    boards = _holdem_root_board_samples()
    probability = 1.0 / len(boards)
    return tuple(
        ChanceOutcome(state=_make_holdem_dealt_state(state, board), probability=probability)
        for board in boards
    )


def _make_holdem_dealt_state(state: GameState, board: Board) -> GameState:
    return GameState(
        board=board,
        players=state.players,
        betting_round=state.betting_round,
        phase=state.phase,
        dealer=state.dealer,
    )


def _holdem_root_board_samples() -> tuple[Board, ...]:
    return (
        Board.from_str("AhKdTc"),
        Board.from_str("QsJh9c"),
        Board.from_str("AsKh7d"),
        Board.from_str("AdQc8s"),
    )


def resolve_game_state_spec(spec: GameStateSpec | None) -> GameStateSpec | None:
    if spec is None:
        return None
    if spec.mode is GameStateMode.EXACT:
        return spec
    if spec.mode is GameStateMode.RANDOM:
        return spec
    raise ValueError(f"unsupported game state mode: {spec.mode}")
