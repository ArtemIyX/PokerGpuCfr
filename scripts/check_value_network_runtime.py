from __future__ import annotations

import os
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from pokergpu.cli import _real_postflop_resolve_spec
from pokergpu.eval import CpuStubLeafEvaluator
from pokergpu.runtime import resolve_postflop_hu
from pokergpu.runtime.value_network import default_postflop_leaf_evaluator


def _strategy_text(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.6f}" for value in values)


def main() -> int:
    checkpoint = os.getenv("POKERGPU_POSTFLOP_VNET_CHECKPOINT", "").strip()
    if not checkpoint:
        raise SystemExit("set POKERGPU_POSTFLOP_VNET_CHECKPOINT")

    path = Path(checkpoint).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")

    os.environ["POKERGPU_POSTFLOP_VNET_CHECKPOINT"] = str(path)
    vnet = default_postflop_leaf_evaluator()
    if isinstance(vnet, CpuStubLeafEvaluator):
        raise SystemExit("value network did not load")

    cpu = CpuStubLeafEvaluator()
    seeds = [1337, 2024, 9001]
    diffs: list[float] = []

    print(f"checkpoint={path}")
    for seed in seeds:
        spec = _real_postflop_resolve_spec(Random(seed))
        result_vnet = resolve_postflop_hu(spec, evaluator=vnet)
        result_cpu = resolve_postflop_hu(spec, evaluator=cpu)

        if not np.isfinite(result_vnet.root_strategy).all():
            raise SystemExit(f"non-finite strategy for seed={seed}")
        if not np.isfinite(result_cpu.root_strategy).all():
            raise SystemExit(f"non-finite cpu strategy for seed={seed}")

        diff = float(np.max(np.abs(result_vnet.root_strategy - result_cpu.root_strategy)))
        diffs.append(diff)
        print(f"seed={seed}")
        print(f"board={spec.state.board}")
        print(f"actions={','.join(result_vnet.root_actions)}")
        print(f"vnet_strategy={_strategy_text(np.asarray(result_vnet.root_strategy))}")
        print(f"cpu_strategy={_strategy_text(np.asarray(result_cpu.root_strategy))}")
        print(f"max_abs_diff={diff:.6f}")
        print(f"leaf_count={result_vnet.leaf_count}")

    if max(diffs) <= 0.0:
        raise SystemExit("value network matches cpu stub exactly, suspicious")

    print("status=ok")
    print(f"max_abs_diff_overall={max(diffs):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
