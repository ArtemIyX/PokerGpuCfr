from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore[assignment]

from pokergpu.abstraction.hands import PlayerRangeVectors
from pokergpu.core.board import Board, Street
from pokergpu.core.cards import Suit
from pokergpu.core.state import GameState
from pokergpu.eval.device import EvalDeviceConfig, resolve_eval_device

_CHIP_SCALE = np.float32(10000.0)


class ValueTargetKind(StrEnum):
    SCALAR_EV = "scalar_ev"
    BUCKETED_VALUE = "bucketed_value"


@dataclass(frozen=True, slots=True)
class PokerValueTarget:
    kind: ValueTargetKind
    player_count: int
    bucket_count: int = 1

    def __post_init__(self) -> None:
        if self.player_count <= 0:
            raise ValueError("player count must be positive")
        if self.kind is ValueTargetKind.SCALAR_EV and self.bucket_count != 1:
            raise ValueError("scalar EV target must use one output per player")
        if self.kind is ValueTargetKind.BUCKETED_VALUE and self.bucket_count <= 1:
            raise ValueError("bucketed target must use more than one bucket")


@dataclass(frozen=True, slots=True)
class ValueFeatureSpec:
    player_count: int
    max_history_length: int = 16
    include_player_mask: bool = True
    include_ranges: bool = True

    def __post_init__(self) -> None:
        if self.player_count <= 0:
            raise ValueError("player count must be positive")
        if self.max_history_length <= 0:
            raise ValueError("max history length must be positive")


@dataclass(frozen=True, slots=True)
class ValueFeatureBatch:
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("feature batch must be two-dimensional")
        if self.values.dtype != np.float32:
            raise ValueError("feature batch must use float32")

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class PokerValueLabel:
    values: NDArray[np.float32]
    kind: ValueTargetKind

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("label batch must be two-dimensional")
        if self.values.dtype != np.float32:
            raise ValueError("label batch must use float32")


def scalar_ev_target(player_count: int) -> PokerValueTarget:
    return PokerValueTarget(kind=ValueTargetKind.SCALAR_EV, player_count=player_count)


def feature_dimension(spec: ValueFeatureSpec) -> int:
    base = 1 + 1 + 5
    if spec.include_player_mask:
        base += spec.player_count
    if spec.include_ranges:
        base += spec.player_count * 1326
    base += spec.player_count
    base += spec.max_history_length
    return base


def _encode_street(board: Board) -> np.float32:
    return np.float32(
        {
            Street.PREFLOP: 0.0,
            Street.FLOP: 1.0,
            Street.TURN: 2.0,
            Street.RIVER: 3.0,
        }[board.street]
    )


def _encode_board(board: Board) -> NDArray[np.float32]:
    encoded = np.zeros(5, dtype=np.float32)
    for index, _card in enumerate(board.cards):
        card = board.cards[index]
        rank_index = float(card.rank.order_value - 2)
        suit_index = float(
            {
                Suit.CLUBS: 0,
                Suit.DIAMONDS: 1,
                Suit.HEARTS: 2,
                Suit.SPADES: 3,
            }[card.suit]
        )
        encoded[index] = np.float32((rank_index * 4.0 + suit_index + 1.0) / 52.0)
    return encoded


def _player_mask(state: GameState) -> NDArray[np.float32]:
    mask = np.zeros(state.player_count, dtype=np.float32)
    mask[int(state.betting_round.to_act)] = np.float32(1.0)
    return mask


def _stack_features(state: GameState) -> NDArray[np.float32]:
    return np.asarray(
        [float(stack.stack) / float(_CHIP_SCALE) for stack in state.betting_round.stacks],
        dtype=np.float32,
    )


def _history_features(state: GameState, max_history_length: int) -> NDArray[np.float32]:
    history = np.zeros(max_history_length, dtype=np.float32)
    history[0] = np.float32(state.player_count)
    history[1] = np.float32(len(state.active_players))
    history[2] = np.float32(len(state.folded_players))
    history[3] = np.float32(int(state.dealer))
    return history


def _range_features(
    ranges: PlayerRangeVectors, 
    player_count: int) -> NDArray[np.float32]:
    values = np.zeros(player_count * 1326, dtype=np.float32)
    for index, range_vector in enumerate(ranges.values[:player_count]):
        start = index * 1326
        values[start : start + 1326] = range_vector.values
    return values


def build_value_feature_batch(
    states: list[GameState] | tuple[GameState, ...],
    ranges: list[PlayerRangeVectors] | tuple[PlayerRangeVectors, ...],
    spec: ValueFeatureSpec,
    device_config: EvalDeviceConfig | None = None,
) -> ValueFeatureBatch:
    if len(states) != len(ranges):
        raise ValueError("states and ranges must have the same length")

    features = np.zeros((len(states), feature_dimension(spec)), dtype=np.float32)
    for row, (state, player_ranges) in enumerate(zip(states, ranges, strict=True)):
        if state.player_count != spec.player_count:
            raise ValueError("state player count must match feature spec")
        if len(player_ranges.values) != spec.player_count:
            raise ValueError("range count must match feature spec")

        offset = 0
        features[row, offset] = _encode_street(state.board)
        offset += 1
        features[row, offset] = np.float32(
            float(sum(stack.stack for stack in state.betting_round.stacks))
            / float(_CHIP_SCALE)
        )
        offset += 1
        features[row, offset : offset + 5] = _encode_board(state.board)
        offset += 5
        if spec.include_player_mask:
            features[row, offset : offset + spec.player_count] = _player_mask(state)
            offset += spec.player_count
        if spec.include_ranges:
            features[row, offset : offset + spec.player_count * 1326] = _range_features(
                player_ranges,
                spec.player_count,
            )
            offset += spec.player_count * 1326
        features[row, offset : offset + spec.player_count] = _stack_features(state)
        offset += spec.player_count
        features[row, offset : offset + spec.max_history_length] = _history_features(
            state,
            spec.max_history_length,
        )
    batch = ValueFeatureBatch(features)
    if device_config is not None and torch is not None:
        device = resolve_eval_device(device_config)
        if device.type == "cuda":
            _ = torch.as_tensor(batch.values, device=device)
    return batch


def build_value_label(
    values: (
        list[float]
        | tuple[float, ...]
        | list[list[float]]
        | tuple[tuple[float, ...], ...]
        | NDArray[np.float32]
    ),
    target: PokerValueTarget,
) -> PokerValueLabel:
    arr = np.asarray(values, dtype=np.float32)
    if target.kind is ValueTargetKind.SCALAR_EV:
        if arr.ndim != 1 or arr.shape[0] != target.player_count:
            raise ValueError("scalar EV labels must match player count")
        return PokerValueLabel(arr.reshape(1, target.player_count), target.kind)
    if arr.ndim != 2 or arr.shape[1] != target.bucket_count:
        raise ValueError("bucketed labels must match bucket count")
    return PokerValueLabel(arr.astype(np.float32, copy=False), target.kind)
