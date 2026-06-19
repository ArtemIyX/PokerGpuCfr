from __future__ import annotations

import pytest

from pokergpu.cfr.solver.spec import CfrVariant
from pokergpu.cfr.solver.spec import GameStateMode
from pokergpu.cfr.solver.spec import GameStateSpec
from pokergpu.cfr.solver.spec import GameVariant
from pokergpu.cfr.solver.spec import ProfilingKind
from pokergpu.cfr.solver.spec import ProfilerSpec
from pokergpu.cfr.solver.spec import SolverStageRequest
from pokergpu.cfr.solver.spec import SolverStageResult
from pokergpu.cfr.solver.spec import TimingSpec


def test_solver_variant_enums_expose_expected_values() -> None:
    assert GameVariant.KUHN.value == "kuhn"
    assert GameVariant.LEDUC.value == "leduc"
    assert GameVariant.HOLDEM_HU.value == "holdem_hu"
    assert GameVariant.HOLDEM_6MAX.value == "holdem_6max"

    assert CfrVariant.CFR.value == "cfr"
    assert CfrVariant.CFR_PLUS.value == "cfr_plus"
    assert CfrVariant.DCFR.value == "dcfr"
    assert CfrVariant.PREDICTIVE_CFR_PLUS.value == "predictive_cfr_plus"

    assert ProfilingKind.CPROFILE.value == "cprofile"
    assert ProfilingKind.TORCH.value == "torch"
    assert ProfilingKind.BOTH.value == "both"

    assert GameStateMode.EXACT.value == "exact"
    assert GameStateMode.RANDOM.value == "random"


def test_game_state_spec_requires_encoded_state_for_exact_mode() -> None:
    with pytest.raises(ValueError, match="exact game state requires encoded_state"):
        GameStateSpec(mode=GameStateMode.EXACT)


def test_game_state_spec_accepts_random_mode_without_encoded_state() -> None:
    spec = GameStateSpec(mode=GameStateMode.RANDOM)

    assert spec.mode is GameStateMode.RANDOM
    assert spec.seed is None
    assert spec.encoded_state is None


def test_solver_stage_request_defaults_are_stable() -> None:
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR,
        depth_limit=3,
    )

    assert request.iterations == 1
    assert request.cpu_workers == 2
    assert request.cpu_workers_stage3 is None
    assert request.cpu_workers_stage4 is None
    assert request.cpu_workers_stage6 is None
    assert request.cpu_workers_stage7 is None
    assert request.effective_cpu_workers_stage3 == 2
    assert request.effective_cpu_workers_stage4 == 2
    assert request.effective_cpu_workers_stage6 == 2
    assert request.effective_cpu_workers_stage7 == 2
    assert request.measure_timing is False
    assert request.timing == TimingSpec()
    assert request.seed is None
    assert request.effective_seed == 0


def test_solver_stage_request_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="depth_limit must be non-negative"):
        SolverStageRequest(
            game=GameVariant.KUHN,
            cfr_variant=CfrVariant.CFR,
            depth_limit=-1,
        )

    with pytest.raises(ValueError, match="iterations must be positive"):
        SolverStageRequest(
            game=GameVariant.KUHN,
            cfr_variant=CfrVariant.CFR,
            depth_limit=1,
            iterations=0,
        )

    with pytest.raises(ValueError, match="cpu_workers must be positive"):
        SolverStageRequest(
            game=GameVariant.KUHN,
            cfr_variant=CfrVariant.CFR,
            depth_limit=1,
            cpu_workers=0,
        )

    with pytest.raises(ValueError, match="per-stage worker counts must be positive"):
        SolverStageRequest(
            game=GameVariant.KUHN,
            cfr_variant=CfrVariant.CFR,
            depth_limit=1,
            cpu_workers_stage3=0,
        )


def test_solver_stage_request_uses_timing_and_seed_overrides() -> None:
    request = SolverStageRequest(
        game=GameVariant.LEDUC,
        cfr_variant=CfrVariant.DCFR,
        depth_limit=2,
        seed=17,
        timing=TimingSpec(measure=True, include_stage_breakdown=False),
    )

    assert request.measure_timing is True
    assert request.effective_seed == 17
    assert request.timing.measure is True
    assert request.timing.include_stage_breakdown is False


def test_solver_stage_result_holds_request_and_optional_outputs() -> None:
    request = SolverStageRequest(
        game=GameVariant.KUHN,
        cfr_variant=CfrVariant.CFR_PLUS,
        depth_limit=4,
    )
    result = SolverStageResult(
        request=request,
        timing_seconds={"stage1": 0.1},
        profiler_output="profile.out",
        diagnostics={"note": "ok"},
    )

    assert result.request == request
    assert result.timing_seconds == {"stage1": 0.1}
    assert result.profiler_output == "profile.out"
    assert result.diagnostics == {"note": "ok"}
