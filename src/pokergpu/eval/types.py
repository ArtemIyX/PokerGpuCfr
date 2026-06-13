from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pokergpu.core.state import GameState


@dataclass(slots=True, frozen=True)
class LeafFeature:
    node_index: int
    player_to_act: int
    street: int
    pot: float
    stack_p0: float
    stack_p1: float
    board_size: int
    reach_p0: float
    reach_p1: float
    reach_p2: float
    is_terminal: bool
    is_frontier: bool
    infoset_id: int | None


@dataclass(slots=True, frozen=True)
class LeafFeatureBatch:
    node_indices: tuple[int, ...]
    node_states: tuple[GameState, ...] | None
    terminal_payoff: NDArray[np.float32]
    player_to_act: NDArray[np.int32]
    street: NDArray[np.int32]
    pot: NDArray[np.float32]
    stack_p0: NDArray[np.float32]
    stack_p1: NDArray[np.float32]
    board_size: NDArray[np.int32]
    reach_p0: NDArray[np.float32]
    reach_p1: NDArray[np.float32]
    reach_p2: NDArray[np.float32]
    is_terminal: NDArray[np.bool_]
    is_frontier: NDArray[np.bool_]
    infoset_id: NDArray[np.int32]

    @property
    def size(self) -> int:
        return len(self.node_indices)


@dataclass(slots=True, frozen=True)
class LeafValueBatch:
    values: NDArray[np.float32]
    ev_player0: NDArray[np.float32]
    ev_player1: NDArray[np.float32]
    ev_player2: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("leaf value matrix must be two-dimensional")
        if self.ev_player0.shape != self.ev_player1.shape:
            raise ValueError("leaf EV arrays must have matching shapes")
        if self.ev_player2 is not None and self.ev_player2.shape != self.ev_player0.shape:
            raise ValueError("leaf EV arrays must have matching shapes")
        if self.values.shape[0] != self.ev_player0.shape[0]:
            raise ValueError("leaf values must match batch size")
