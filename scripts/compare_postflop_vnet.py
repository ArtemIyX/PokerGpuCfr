from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"torch is required: {exc}") from exc

from pokergpu.abstraction.hands import RangeVector
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board, Street
from pokergpu.core.cards import Card, make_deck
from pokergpu.core.state import GameState, PlayerState
from pokergpu.eval.cpu_stub import CpuStubLeafEvaluator
from pokergpu.eval.interface import LeafEvaluator
from pokergpu.eval.types import LeafFeatureBatch, LeafValueBatch
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.value_network.checkpoint import load_checkpoint
from pokergpu.value_network.dataset import FeatureNormalizer, normalize_feature_batch
from pokergpu.value_network.dataset import ValueFeatureBatch
from pokergpu.value_network.model import build_value_model, infer_value
from pokergpu.value_network.target import ValueFeatureSpec, ValueTargetKind


@dataclass(frozen=True, slots=True)
class ResultRow:
    label: str
    elapsed_s: float
    ev_p0: float
    ev_p1: float
    root_strategy: str
    iterations: int
    nodes: int
    leaves: int
    depth: int


class CheckpointLeafEvaluator(LeafEvaluator):
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        checkpoint, model_state, _optimizer_state = load_checkpoint(checkpoint_path)
        if checkpoint.target_kind is not ValueTargetKind.SCALAR_EV:
            raise ValueError("checkpoint must use scalar EV targets")
        if checkpoint.feature_spec.player_count != 2:
            raise ValueError("script currently supports heads-up checkpoints only")
        model = build_value_model(checkpoint.model_config, device=device)
        model.load_state_dict(model_state, strict=True)
        self._model = model
        self._feature_spec = checkpoint.feature_spec
        self._normalizer = checkpoint.normalizer

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        features = _build_runtime_leaf_features(batch, self._feature_spec)
        normalized = normalize_feature_batch(
            ValueFeatureBatch(features),
            self._normalizer,
        ).values
        values = infer_value(self._model, normalized)
        if values.shape != (batch.size, 2):
            raise ValueError("checkpoint model must return two outputs")
        return LeafValueBatch(
            ev_player0=np.asarray(values[:, 0], dtype=np.float32),
            ev_player1=np.asarray(values[:, 1], dtype=np.float32),
        )


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    rng = Random(args.seed)
    state = make_random_state(rng, args.players, args.street)
    range_p0, range_p1 = make_random_ranges(rng)

    reference = run_case(
        "cpu_stub",
        state,
        range_p0,
        range_p1,
        args.time_budget,
        args.max_depth,
        args.max_nodes,
        CpuStubLeafEvaluator(),
    )

    checkpoint_eval = CheckpointLeafEvaluator(args.checkpoint, args.device)
    checkpoint = run_case(
        "checkpoint",
        state,
        range_p0,
        range_p1,
        args.time_budget,
        args.max_depth,
        args.max_nodes,
        checkpoint_eval,
    )

    print_case_summary(args, state, reference, checkpoint)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--street", choices=("flop", "turn", "river"), default="flop")
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def make_random_state(rng: Random, players: int, street_name: str) -> GameState:
    if players != 2:
        raise SystemExit("current resolver is heads-up only; use --players 2")
    deck = list(make_deck())
    rng.shuffle(deck)
    board_cards = {"flop": 3, "turn": 4, "river": 5}[street_name]
    board = tuple(deck[:board_cards])
    hole0 = (deck[board_cards], deck[board_cards + 1])
    hole1 = (deck[board_cards + 2], deck[board_cards + 3])
    street = Board(board).street
    stacks = (
        PlayerStack(player=PlayerIndex(0), stack=chips(2000)),
        PlayerStack(player=PlayerIndex(1), stack=chips(2000)),
    )
    bets = (
        PlayerBet(player=PlayerIndex(0), committed=chips(0)),
        PlayerBet(player=PlayerIndex(1), committed=chips(0)),
    )
    state = GameState(
        board=Board(cards=board),
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=hole0),
            PlayerState(player=PlayerIndex(1), hole_cards=hole1),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(200)),
            stacks=stacks,
            bets=bets,
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )
    if state.current_street is not street:
        raise RuntimeError("generated street mismatch")
    return state


def make_random_ranges(rng: Random) -> tuple[RangeVector, RangeVector]:
    values_p0 = np.asarray([rng.random() for _ in range(1326)], dtype=np.float32)
    values_p1 = np.asarray([rng.random() for _ in range(1326)], dtype=np.float32)
    return RangeVector.from_values(values_p0).normalized(), RangeVector.from_values(values_p1).normalized()


def run_case(
    label: str,
    state: GameState,
    range_p0: RangeVector,
    range_p1: RangeVector,
    time_budget: float,
    max_depth: int,
    max_nodes: int,
    evaluator: LeafEvaluator,
) -> ResultRow:
    started = time.perf_counter()
    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=state,
            range_p0=range_p0,
            range_p1=range_p1,
            time_budget_sec=time_budget,
            max_depth=max_depth,
            max_nodes=max_nodes,
        ),
        evaluator=evaluator,
    )
    elapsed = time.perf_counter() - started
    return ResultRow(
        label=label,
        elapsed_s=elapsed,
        ev_p0=float(result.root_ev_player0),
        ev_p1=float(result.root_ev_player1),
        root_strategy=np.array2string(result.root_strategy, precision=4, separator=","),
        iterations=int(result.iterations),
        nodes=int(result.node_count),
        leaves=int(result.leaf_count),
        depth=max_depth,
    )


def print_case_summary(
    args: argparse.Namespace,
    state: GameState,
    reference: ResultRow,
    checkpoint: ResultRow,
) -> None:
    delta_p0 = checkpoint.ev_p0 - reference.ev_p0
    delta_p1 = checkpoint.ev_p1 - reference.ev_p1
    rows = [
        ("seed", str(args.seed)),
        ("players", str(args.players)),
        ("street", str(args.street)),
        ("device", args.device),
        ("board", str(state.board)),
        ("p0_hole", f"{state.players[0].hole_cards[0]}{state.players[0].hole_cards[1]}"),
        ("p1_hole", f"{state.players[1].hole_cards[0]}{state.players[1].hole_cards[1]}"),
        ("reference", reference.label),
        ("reference_ev", f"{reference.ev_p0:.6f} / {reference.ev_p1:.6f}"),
        ("reference_time_s", f"{reference.elapsed_s:.6f}"),
        ("checkpoint", checkpoint.label),
        ("checkpoint_ev", f"{checkpoint.ev_p0:.6f} / {checkpoint.ev_p1:.6f}"),
        ("checkpoint_time_s", f"{checkpoint.elapsed_s:.6f}"),
        ("delta_ev", f"{delta_p0:.6f} / {delta_p1:.6f}"),
        ("iterations", str(checkpoint.iterations)),
        ("nodes", str(checkpoint.nodes)),
        ("leaves", str(checkpoint.leaves)),
        ("depth", str(checkpoint.depth)),
        ("root_strategy", checkpoint.root_strategy),
    ]
    width = max(len(key) for key, _value in rows)
    for key, value in rows:
        print(f"{key:<{width}} : {value}")


def _build_runtime_leaf_features(
    batch: LeafFeatureBatch,
    feature_spec: ValueFeatureSpec,
) -> np.ndarray:
    feature_count = (
        1
        + 1
        + 5
        + feature_spec.player_count
        + (feature_spec.player_count * 1326)
        + feature_spec.player_count
        + feature_spec.max_history_length
    )
    features = np.zeros((batch.size, feature_count), dtype=np.float32)
    offset = 0
    features[:, offset] = np.asarray(batch.street, dtype=np.float32)
    offset += 1
    features[:, offset] = np.asarray(batch.pot, dtype=np.float32)
    offset += 1
    offset += 5
    for row_index, player_index in enumerate(np.asarray(batch.player_to_act, dtype=np.int32)):
        if 0 <= player_index < feature_spec.player_count:
            features[row_index, offset + player_index] = 1.0
    offset += feature_spec.player_count
    offset += feature_spec.player_count * 1326
    if feature_spec.player_count >= 1:
        features[:, offset] = np.asarray(batch.stack_p0, dtype=np.float32)
    if feature_spec.player_count >= 2:
        features[:, offset + 1] = np.asarray(batch.stack_p1, dtype=np.float32)
    offset += feature_spec.player_count
    features[:, offset + 0] = np.asarray(batch.board_size, dtype=np.float32)
    features[:, offset + 1] = np.asarray(batch.player_to_act, dtype=np.float32)
    features[:, offset + 2] = np.asarray(batch.is_terminal.astype(np.float32), dtype=np.float32)
    features[:, offset + 3] = np.asarray(batch.is_frontier.astype(np.float32), dtype=np.float32)
    features[:, offset + 4] = np.asarray(np.clip(batch.infoset_id, -1, 1_000_000), dtype=np.float32)
    return features


if __name__ == "__main__":
    raise SystemExit(main())
