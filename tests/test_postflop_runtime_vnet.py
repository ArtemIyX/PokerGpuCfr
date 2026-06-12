from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
from pokergpu.core.board import Board
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.value_network.checkpoint import ValueCheckpoint, save_checkpoint
from pokergpu.value_network.dataset import FeatureNormalizer
from pokergpu.value_network.model import (
    ValueNetworkConfig,
    build_value_model,
)
from pokergpu.value_network.target import ValueFeatureSpec, ValueTargetKind


def _make_state() -> GameState:
    return GameState(
        board=Board.from_str("AhKdTc"),
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )


def test_postflop_resolver_uses_default_value_network_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    spec = ValueFeatureSpec(player_count=2, max_history_length=10)
    config = ValueNetworkConfig(
        input_dim=10,
        hidden_dim=32,
        hidden_layers=2,
        output_dim=2,
    )
    model = build_value_model(config, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "postflop_vnet.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        ValueCheckpoint(
            step=1,
            best_metric=0.0,
            feature_spec=spec,
            target_kind=ValueTargetKind.SCALAR_EV,
            target_bucket_count=1,
            model_config=config,
            normalizer=FeatureNormalizer(
                mean=np.zeros(config.input_dim, dtype=np.float32),
                std=np.ones(config.input_dim, dtype=np.float32),
            ),
        ),
    )
    monkeypatch.setenv("POKERGPU_POSTFLOP_VNET_CHECKPOINT", str(checkpoint_path))

    result = resolve_postflop_hu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.root_infoset_id == 0
    assert np.isclose(float(result.root_strategy.sum()), 1.0)
    assert result.iterations == 1
    assert result.node_count > 0
    assert result.leaf_count > 0


def test_postflop_gpu_resolver_returns_coherent_root_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")

    from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu

    result = resolve_postflop_gpu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert result.root_infoset_id == 0
    assert np.isclose(float(result.root_strategy.sum()), 1.0)
    assert result.root_action_ev_player0.shape == result.root_strategy.shape
    assert result.root_action_ev_player1.shape == result.root_strategy.shape
    assert np.isclose(
        result.root_ev_player0,
        float(np.sum(result.root_strategy * result.root_action_ev_player0) / 300.0),
    )
    assert np.isclose(result.root_ev_player1, -result.root_ev_player0)


def test_postflop_resolver_falls_back_deterministically_when_checkpoint_is_bad(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_checkpoint = tmp_path / "missing.pt"
    monkeypatch.setenv("POKERGPU_POSTFLOP_VNET_CHECKPOINT", str(bad_checkpoint))

    result_a = resolve_postflop_hu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )
    result_b = resolve_postflop_hu(
        PostflopResolveSpec(
            state=_make_state(),
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            max_depth=1,
            max_nodes=16,
        )
    )

    assert np.allclose(result_a.root_strategy, result_b.root_strategy)
    assert result_a.root_infoset_id == result_b.root_infoset_id
    assert result_a.iterations == 1
    assert result_b.iterations == 1
