from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Protocol

import numpy as np

from .spec import DebugSpec


class SolverDebugSink(Protocol):
    def add_scalar(self, tag: str, value: float, step: int) -> None: ...

    def add_histogram(self, tag: str, values: np.ndarray | Sequence[float], step: int) -> None: ...

    def add_text(self, tag: str, text: str, step: int) -> None: ...

    def add_sample(self, tag: str, values: Sequence[float], step: int, limit: int) -> None: ...

    def flush(self) -> None: ...


@dataclass(slots=True)
class NoopDebugSink:
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        _ = tag, value, step

    def add_histogram(self, tag: str, values: np.ndarray | Sequence[float], step: int) -> None:
        _ = tag, values, step

    def add_text(self, tag: str, text: str, step: int) -> None:
        _ = tag, text, step

    def add_sample(self, tag: str, values: Sequence[float], step: int, limit: int) -> None:
        _ = tag, values, step, limit

    def flush(self) -> None:
        return None


@dataclass(slots=True)
class DebugSession:
    spec: DebugSpec
    sink: SolverDebugSink
    log_dir: Path | None = None

    def close(self) -> None:
        self.sink.flush()


def create_debug_session(spec: DebugSpec, *, run_name: str) -> DebugSession:
    if not spec.enabled:
        return DebugSession(spec=spec, sink=NoopDebugSink())

    log_dir = spec.log_dir or _default_log_dir(run_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    session = DebugSession(spec=spec, sink=NoopDebugSink(), log_dir=log_dir)
    sink = session.sink
    sink.add_text("debug/run_name", run_name, 0)
    sink.add_text("debug/log_dir", str(log_dir), 0)
    return session


def _default_log_dir(run_name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    pid = os.getpid()
    return Path("artifacts") / "debug" / f"{run_name}-{stamp}-pid{pid}"


def _format_sequence(values: Iterable[float]) -> str:
    return ", ".join(f"{float(value):.6g}" for value in values)


def summarize_array(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0.0}
    finite = np.isfinite(array)
    abs_array = np.abs(array)
    summary = {
        "count": float(array.size),
        "finite_count": float(np.count_nonzero(finite)),
        "nan_count": float(np.count_nonzero(np.isnan(array))),
        "inf_count": float(np.count_nonzero(np.isinf(array))),
        "zero_count": float(np.count_nonzero(array == 0.0)),
        "nonzero_count": float(np.count_nonzero(array != 0.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "abs_min": float(np.min(abs_array)),
        "abs_max": float(np.max(abs_array)),
        "l1": float(np.sum(abs_array)),
        "l2": float(np.sqrt(np.sum(np.square(array)))),
    }
    return summary


def log_summary(sink: SolverDebugSink, prefix: str, values: Sequence[float] | np.ndarray, step: int) -> None:
    array = np.asarray(values, dtype=np.float64)
    stats = summarize_array(array)
    for key, value in stats.items():
        sink.add_scalar(f"{prefix}/{key}", value, step)
    sink.add_histogram(f"{prefix}/hist", array, step)
    sink.add_sample(f"{prefix}/sample", array.tolist(), step, limit=16)
    if array.size > 0:
        sink.add_text(
            f"{prefix}/preview",
            _format_sequence(array[: min(8, array.size)]),
            step,
        )


def log_text_map(sink: SolverDebugSink, prefix: str, values: dict[str, object], step: int, *, limit: int) -> None:
    items = list(values.items())[:limit]
    text = "\n".join(f"{key}: {value}" for key, value in items)
    sink.add_text(prefix, text, step)
