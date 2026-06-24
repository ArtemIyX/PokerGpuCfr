from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass

from pokergpu.abstraction.actions import BaselineActionAbstraction
from pokergpu.abstraction.actions import make_compact_profile
from pokergpu.abstraction.actions import make_holdem_hu_profile
from pokergpu.core.board import Board
from pokergpu.core.board import Street
from pokergpu.core.betting import BettingRoundState
from pokergpu.core.betting import BlindStructure
from pokergpu.core.betting import PlayerBet
from pokergpu.core.betting import PlayerIndex
from pokergpu.core.betting import PlayerStack
from pokergpu.core.betting import Pot
from pokergpu.core.betting import Chips
from pokergpu.core.betting import chips
from pokergpu.core.cards import Card
from pokergpu.core.cards import make_deck
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


@dataclass(slots=True, frozen=True)
class HoldemHuTreeConfig:
    compact: bool = False
    board_sample_limit: int = 8


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
    state: GameState | None = None,
    holdem_hu_tree_config: HoldemHuTreeConfig | None = None,
) -> PublicTree:
    if game is GameVariant.KUHN:
        return make_kuhn_public_tree()
    if game is GameVariant.LEDUC:
        return make_leduc_public_tree()
    if game is GameVariant.HOLDEM_HU:
        return make_holdem_hu_public_tree(
            compact=compact or (depth_limit is not None and depth_limit <= 1),
            state=state,
            config=holdem_hu_tree_config,
        )
    supported = ", ".join(
        (
            GameVariant.KUHN.value,
            GameVariant.LEDUC.value,
            GameVariant.HOLDEM_HU.value,
        )
    )
    raise NotImplementedError(f"unsupported game variant {game.value!r}; supported variants: {supported}")


def make_holdem_hu_public_tree(
    *,
    compact: bool = False,
    state: GameState | None = None,
    config: HoldemHuTreeConfig | None = None,
) -> PublicTree:
    state = state or _make_canonical_holdem_state()
    tree_config = config or HoldemHuTreeConfig(compact=compact)
    effective_compact = compact or tree_config.compact
    abstraction = BaselineActionAbstraction(
        profile=make_compact_profile() if effective_compact else make_holdem_hu_profile()
    )
    build_config = TreeBuildConfig(max_depth=1 if effective_compact else 6, max_nodes=256 if effective_compact else 1024)
    expand_chance = None
    if not effective_compact:
        def expand_chance(current_state: GameState) -> tuple[ChanceOutcome, ...] | None:
            if not current_state.board.is_preflop:
                return None
            return _expand_holdem_chance(
                current_state,
                board_sample_limit=tree_config.board_sample_limit,
            )
    return build_public_tree(
        state,
        abstraction=abstraction,
        config=build_config,
        expand_chance=expand_chance,
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


def _expand_holdem_chance(state: GameState, *, board_sample_limit: int = 8) -> tuple[ChanceOutcome, ...] | None:
    if state.phase is not HandPhase.IN_PROGRESS:
        return None
    if state.board.is_preflop:
        boards = _sample_public_boards(Street.FLOP, limit=board_sample_limit)
    elif state.board.is_flop:
        boards = _sample_public_boards(Street.TURN, prefix=state.board.cards, limit=board_sample_limit)
    elif state.board.is_turn:
        boards = _sample_public_boards(Street.RIVER, prefix=state.board.cards, limit=board_sample_limit)
    else:
        return None
    if not boards:
        return None
    probability = 1.0 / len(boards)
    return tuple(ChanceOutcome(state=_make_holdem_dealt_state(state, board), probability=probability) for board in boards)


def _make_holdem_dealt_state(state: GameState, board: Board) -> GameState:
    public_players = tuple(
        PlayerState(
            player=player.player,
            hole_cards=None,
            folded=player.folded,
            all_in=player.all_in,
        )
        for player in state.players
    )
    return GameState(
        board=board,
        players=public_players,
        betting_round=state.betting_round,
        phase=state.phase,
        dealer=state.dealer,
    )


def _sample_public_boards(street: Street, *, prefix: tuple[Card, ...] = (), limit: int = 4) -> tuple[Board, ...]:
    deck = make_deck()
    dead_cards = set(prefix)
    target_len = {
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }[street]
    if len(prefix) > target_len:
        return ()
    if len(prefix) == target_len:
        return (Board(cards=prefix),)
    remaining_cards = [card for card in deck if card not in dead_cards]
    needed_cards = target_len - len(prefix)
    if needed_cards <= 0:
        return (Board(cards=prefix),)
    boards: list[Board] = []
    for extra_cards in _spread_board_samples(remaining_cards, needed_cards, limit):
        boards.append(Board(cards=prefix + extra_cards))
    return tuple(boards)


def _spread_board_samples(cards: list[Card], needed_cards: int, limit: int) -> tuple[tuple[Card, ...], ...]:
    if limit <= 0:
        return ()
    if needed_cards == 1:
        selected_indices = _spread_indices(len(cards), limit)
        return tuple((cards[index],) for index in selected_indices)
    if needed_cards == 2:
        pairs: list[tuple[Card, Card]] = []
        selected_indices = _spread_indices(len(cards), min(len(cards), max(2, limit * 2)))
        for left_index, left in enumerate(selected_indices):
            for right in selected_indices[left_index + 1 :]:
                pairs.append((cards[left], cards[right]))
                if len(pairs) >= limit:
                    return tuple(pairs)
        if pairs:
            return tuple(pairs)
    if needed_cards == 3:
        triples: list[tuple[Card, Card, Card]] = []
        selected_indices = _spread_indices(len(cards), min(len(cards), max(3, limit * 2)))
        for first_index, first in enumerate(selected_indices):
            for second_index, second in enumerate(selected_indices[first_index + 1 :], start=first_index + 1):
                for third in selected_indices[second_index + 1 :]:
                    triples.append((cards[first], cards[second], cards[third]))
                    if len(triples) >= limit:
                        return tuple(triples)
        if triples:
            return tuple(triples)
    return tuple(combinations(cards, needed_cards))[:limit]


def _spread_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if count <= 1:
        return (0,)
    if count >= length:
        return tuple(range(length))
    step = (length - 1) / float(count - 1)
    indices: list[int] = []
    seen: set[int] = set()
    for position in range(count):
        index = int(round(position * step))
        index = min(length - 1, max(0, index))
        while index in seen and index + 1 < length:
            index += 1
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return tuple(indices)


def resolve_game_state_spec(spec: GameStateSpec | None) -> GameStateSpec | None:
    if spec is None:
        return None
    if spec.mode is GameStateMode.EXACT:
        return spec
    if spec.mode is GameStateMode.RANDOM:
        return spec
    raise ValueError(f"unsupported game state mode: {spec.mode}")
