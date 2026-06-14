from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "single_spot_postflop_solve.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    base_env = os.environ.copy()
    uncompiled = _run_case(base_env, False)
    compiled = _run_case(base_env, True)

    print(json.dumps({
        "uncompiled": uncompiled,
        "compiled": compiled,
        "delta_seconds": uncompiled["timing"]["elapsed_seconds"] - compiled["timing"]["elapsed_seconds"],
    }, indent=2))
    return 0


def _run_case(env: dict[str, str], enabled: bool) -> dict[str, object]:
    case_env = env.copy()
    case_env["POKERGPU_COMPACT_COMPILE"] = "1" if enabled else "0"
    proc = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else Path(sys.executable)),
            str(SCRIPT),
            "--board",
            "AhKdTc",
            "--iterations",
            "8",
            "--max-depth",
            "5",
            "--max-nodes",
            "4096",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=case_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"solver failed with exit code {proc.returncode}")
    return json.loads(proc.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
