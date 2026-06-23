from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path

from .state import DenseCfrState
from .state import SolverState


class GameVariant(StrEnum):
    KUHN = "kuhn"
    LEDUC = "leduc"
    HOLDEM_HU = "holdem_hu"
    HOLDEM_6MAX = "holdem_6max"


class CfrVariant(StrEnum):
    CFR = "cfr"
    CFR_PLUS = "cfr_plus"
    DCFR = "dcfr"
    PREDICTIVE_CFR_PLUS = "predictive_cfr_plus"


class ProfilingKind(StrEnum):
    CPROFILE = "cprofile"
    TORCH = "torch"
    BOTH = "both"


class GameStateMode(StrEnum):
    EXACT = "exact"
    RANDOM = "random"


@dataclass(slots=True, frozen=True)
class GameStateSpec:
    mode: GameStateMode
    seed: int | None = None
    encoded_state: bytes | None = None

    def __post_init__(self) -> None:
        if self.mode is GameStateMode.EXACT and self.encoded_state is None:
            raise ValueError("exact game state requires encoded_state")


@dataclass(slots=True, frozen=True)
class ProfilerSpec:
    kind: ProfilingKind
    output_path: str | None = None
    record_shapes: bool = False
    with_stack: bool = False


@dataclass(slots=True, frozen=True)
class TimingSpec:
    measure: bool = False
    include_stage_breakdown: bool = True
    include_branch_breakdown: bool = True


@dataclass(slots=True, frozen=True)
class DebugSpec:
    enabled: bool = False
    log_dir: Path | None = None
    sample_limit: int = 64
    histogram_bins: int = 64
    max_text_items: int = 128

    def __post_init__(self) -> None:
        if self.sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if self.max_text_items <= 0:
            raise ValueError("max_text_items must be positive")


@dataclass(slots=True, frozen=True)
class SolverStageRequest:
    game: GameVariant
    cfr_variant: CfrVariant
    depth_limit: int
    iterations: int = 1
    batch_size: int = 1
    state: GameStateSpec | None = None
    seed: int | None = None
    cpu_workers: int = 2
    cpu_workers_stage3: int | None = None
    cpu_workers_stage4: int | None = None
    cpu_workers_stage6: int | None = None
    cpu_workers_stage7: int | None = None
    gpu_backend: str | None = None
    profiler: ProfilerSpec | None = None
    timing: TimingSpec = field(default_factory=TimingSpec)
    debug: DebugSpec = field(default_factory=DebugSpec)
    measure_time: bool = False

    def __post_init__(self) -> None:
        if self.depth_limit < 0:
            raise ValueError("depth_limit must be non-negative")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.cpu_workers <= 0:
            raise ValueError("cpu_workers must be positive")
        for value in (
            self.cpu_workers_stage3,
            self.cpu_workers_stage4,
            self.cpu_workers_stage6,
            self.cpu_workers_stage7,
        ):
            if value is not None and value <= 0:
                raise ValueError("per-stage worker counts must be positive")

    @property
    def effective_cpu_workers_stage3(self) -> int:
        return self.cpu_workers_stage3 or self.cpu_workers

    @property
    def effective_cpu_workers_stage4(self) -> int:
        return self.cpu_workers_stage4 or self.cpu_workers

    @property
    def effective_cpu_workers_stage6(self) -> int:
        return self.cpu_workers_stage6 or self.cpu_workers

    @property
    def effective_cpu_workers_stage7(self) -> int:
        return self.cpu_workers_stage7 or self.cpu_workers

    @property
    def measure_timing(self) -> bool:
        return self.measure_time or self.timing.measure

    @property
    def effective_seed(self) -> int:
        return 0 if self.seed is None else self.seed


@dataclass(slots=True, frozen=True)
class SolverStageResult:
    request: SolverStageRequest
    final_state: SolverState | DenseCfrState | None = None
    timing_seconds: dict[str, float] | None = None
    profiler_output: str | None = None
    diagnostics: dict[str, object] | None = None
