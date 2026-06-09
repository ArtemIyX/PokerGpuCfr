from pokergpu.benchmarks import BenchmarkResult, run_benchmark


def test_run_benchmark_returns_result() -> None:
    result = run_benchmark("noop", lambda: None, iterations=5)

    assert isinstance(result, BenchmarkResult)
    assert result.name == "noop"
    assert result.iterations == 5
    assert result.total_seconds >= 0.0
    assert result.seconds_per_iteration >= 0.0
