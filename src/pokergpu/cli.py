import logging
import sys

from .app import create_app
from .benchmarks import run_benchmark


def main() -> int:
    settings = create_app()
    logger = logging.getLogger(__name__)
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        result = run_benchmark("noop", lambda: None)
        print(
            "benchmark="
            f"{result.name} iterations={result.iterations} "
            f"seconds={result.total_seconds:.6f} "
            f"per_iter={result.seconds_per_iteration:.9f}"
        )
        return 0
    logger.info("PokerGPU initialized")
    print(f"PokerGPU ready on device={settings.device}")
    return 0
