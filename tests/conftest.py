import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run benchmark tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark_suite: marks benchmark tests that are skipped by default",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-benchmarks"):
        return

    skip_benchmark = pytest.mark.skip(reason="use --run-benchmarks to run benchmarks")
    for item in items:
        if "benchmark_suite" in item.keywords:
            item.add_marker(skip_benchmark)
