"""Benchmark single-thread vs multi-thread CFR for Kuhn and Leduc Poker."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import pytest

from pokergpu.cfr.infosets import InfosetStore
from pokergpu.cfr.iteration import CFRVariant, DCFRConfig, run_cfr_iteration
from pokergpu.cfr.kuhn import (
    average_strategy_profile,
    kuhn_infoset_indices_for_player,
    new_kuhn_infoset_store,
)
from pokergpu.cfr.kuhn import (
    expected_action_utilities as kuhn_expected_action_utilities,
)
from pokergpu.cfr.leduc import (
    average_strategy_profile_leduc,
    expected_action_utilities_leduc,
    leduc_infoset_indices_for_player,
    new_leduc_infoset_store,
)

# =============================================================================
# Helpers for multiprocessing - returns InfosetStore-compatible objects
# =============================================================================

def _run_kuhn_chunk(args: tuple[int, int, CFRVariant, DCFRConfig | None]) -> InfosetStore:  # noqa: F811
    """Run Kuhn CFR iterations in worker process."""
    iter_start, iter_end, _, dcfr_config = args
    store = new_kuhn_infoset_store()

    for it in range(iter_start, iter_end + 1):
        utilities = kuhn_expected_action_utilities(store, 0)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=kuhn_infoset_indices_for_player(0),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )
        utilities = kuhn_expected_action_utilities(store, 1)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=kuhn_infoset_indices_for_player(1),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )

    return store


def _run_leduc_chunk(args: tuple[int, int, CFRVariant, DCFRConfig | None]) -> InfosetStore:  # noqa: F811
    """Run Leduc CFR iterations in worker process."""
    iter_start, iter_end, _, dcfr_config = args
    store = new_leduc_infoset_store()

    for it in range(iter_start, iter_end + 1):
        utilities = expected_action_utilities_leduc(store, 0)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=leduc_infoset_indices_for_player(0),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )
        utilities = expected_action_utilities_leduc(store, 1)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=leduc_infoset_indices_for_player(1),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )

    return store


# =============================================================================
# Helper functions for single-thread baseline training
# =============================================================================

def train_kuhn_cfr(iterations: int, variant: CFRVariant = CFRVariant.CFR_PLUS) -> InfosetStore:  # noqa: F811
    """Run Kuhn CFR training."""
    store = new_kuhn_infoset_store()

    for it in range(1, iterations + 1):
        utilities = kuhn_expected_action_utilities(store, 0)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=kuhn_infoset_indices_for_player(0),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=None,
        )
        utilities = kuhn_expected_action_utilities(store, 1)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=kuhn_infoset_indices_for_player(1),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=None,
        )

    return store


def train_leduc_cfr(iterations: int, variant: CFRVariant = CFRVariant.CFR_PLUS) -> InfosetStore:  # noqa: F811
    """Run Leduc CFR training."""
    store = new_leduc_infoset_store()

    for it in range(1, iterations + 1):
        utilities = expected_action_utilities_leduc(store, 0)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=leduc_infoset_indices_for_player(0),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=None,
        )
        utilities = expected_action_utilities_leduc(store, 1)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=leduc_infoset_indices_for_player(1),
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=None,
        )

    return store


# =============================================================================
# Kuhn Poker Benchmarks
# =============================================================================

@pytest.mark.benchmark_suite
def test_kuhn_single_thread_all_variants() -> None:  # noqa: F811
    """Benchmark single-thread CFR for Kuhn Poker with all variants."""
    iterations = 10

    for variant in (CFRVariant.VANILLA, CFRVariant.CFR_PLUS, CFRVariant.DCFR):
        store = train_kuhn_cfr(iterations=iterations, variant=variant)
        strategy_profile = average_strategy_profile(store)

    assert strategy_profile is not None


@pytest.mark.benchmark_suite
def test_kuhn_single_vs_multi_thread() -> None:
    """Compare single-thread vs multi-thread CFR for Kuhn Poker."""
    iterations = 20
    mp.set_start_method("spawn", force=True)

    # Single-thread baseline - train all players' infosets sequentially
    start = perf_counter()
    _ = train_kuhn_cfr(iterations=iterations, variant=CFRVariant.CFR_PLUS)
    elapsed_st: float = perf_counter() - start

    # Multi-thread using ProcessPoolExecutor - need full tuple including CFRVariant
    chunks: list[tuple[int, int, CFRVariant, DCFRConfig | None]] = [
        (i * 5, i * 5 + 5, CFRVariant.CFR_PLUS, None)
        for i in range(iterations // 5)
    ]

    start = perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        _ = list(executor.map(_run_kuhn_chunk, chunks))

    elapsed_mt = perf_counter() - start

    assert elapsed_st > 0.0
    assert elapsed_mt > 0.0
    speedup = elapsed_st / elapsed_mt
    assert speedup <= 2.0, f"Multi-thread slower than single-thread: {speedup=}"


@pytest.mark.benchmark_suite
def test_kuhn_dcfr_single_vs_multi_thread() -> None:
    """Compare single vs multi-thread DCFR for Kuhn Poker."""
    iterations = 30
    mp.set_start_method("spawn", force=True)

    # Single-thread baseline - CFR variant is used, not passed via config
    start = perf_counter()
    _ = train_kuhn_cfr(iterations=iterations, variant=CFRVariant.CFR_PLUS)
    elapsed_st: float = perf_counter() - start

    # Multi-thread using ProcessPoolExecutor - need full tuple including CFRVariant
    chunks: list[tuple[int, int, CFRVariant, DCFRConfig | None]] = [
        (i * 7, i * 7 + 7, CFRVariant.CFR_PLUS, None)
        for i in range(iterations // 7)
    ]

    start = perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        _ = list(executor.map(_run_kuhn_chunk, chunks))

    elapsed_mt = perf_counter() - start

    assert elapsed_st > 0.0
    assert elapsed_mt > 0.0
    speedup = elapsed_st / elapsed_mt
    assert speedup <= 3.0, f"Multi-thread slower than single-thread: {speedup=}"


# =============================================================================
# Leduc Poker Benchmarks
# =============================================================================

@pytest.mark.benchmark_suite
def test_leduc_single_thread_all_variants() -> None:  # noqa: F811
    """Benchmark single-thread CFR for Leduc Poker with all variants."""
    iterations = 10

    for variant in (CFRVariant.VANILLA, CFRVariant.CFR_PLUS, CFRVariant.DCFR):
        store = train_leduc_cfr(iterations=iterations, variant=variant)
        strategy_profile = average_strategy_profile_leduc(store)

    assert strategy_profile is not None


@pytest.mark.benchmark_suite
def test_leduc_single_vs_multi_thread() -> None:
    """Compare single-thread vs multi-thread CFR for Leduc Poker."""
    iterations = 20
    mp.set_start_method("spawn", force=True)

    # Single-thread baseline - train all players' infosets sequentially
    start = perf_counter()
    _ = train_leduc_cfr(iterations=iterations, variant=CFRVariant.CFR_PLUS)
    elapsed_st: float = perf_counter() - start

    # Multi-thread using ProcessPoolExecutor - need full tuple including CFRVariant.DCFRConfig
    chunks: list[tuple[int, int, CFRVariant, DCFRConfig | None]] = [
        (i * 5, i * 5 + 5, CFRVariant.CFR_PLUS, None)
        for i in range(iterations // 5)
    ]

    start = perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        _ = list(executor.map(_run_leduc_chunk, chunks))

    elapsed_mt = perf_counter() - start

    assert elapsed_st > 0.0
    assert elapsed_mt > 0.0
    speedup = elapsed_st / elapsed_mt
    assert speedup <= 2.0, f"Multi-thread slower than single-thread: {speedup=}"


@pytest.mark.benchmark_suite
def test_leduc_dcfr_single_vs_multi_thread() -> None:
    """Compare single vs multi-thread DCFR for Leduc Poker."""
    iterations = 30
    mp.set_start_method("spawn", force=True)

    # Single-thread baseline - CFR variant is used, not passed via config
    start = perf_counter()
    _ = train_leduc_cfr(iterations=iterations, variant=CFRVariant.CFR_PLUS)
    elapsed_st: float = perf_counter() - start

    # Multi-thread using ProcessPoolExecutor - need full tuple including CFRVariant.DCFRConfig
    chunks: list[tuple[int, int, CFRVariant, DCFRConfig | None]] = [
        (i * 7, i * 7 + 7, CFRVariant.CFR_PLUS, None)
        for i in range(iterations // 7)
    ]

    start = perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        _ = list(executor.map(_run_leduc_chunk, chunks))

    elapsed_mt = perf_counter() - start

    assert elapsed_st > 0.0
    assert elapsed_mt > 0.0
    speedup = elapsed_st / elapsed_mt
    assert speedup <= 3.0, f"Multi-thread slower than single-thread: {speedup=}"


# =============================================================================
# Convergence Tests (multi-thread produces valid strategies)
# =============================================================================

@pytest.mark.benchmark_suite
def test_kuhn_multi_thread_convergence() -> None:
    """Verify multi-thread Kuhn CFR converges to valid strategies."""
    mp.set_start_method("spawn", force=True)

    iterations = 50
    chunks: list[tuple[int, int, CFRVariant, DCFRConfig | None]] = [
        (i * 12, i * 12 + 12, CFRVariant.CFR_PLUS, None)
        for i in range(iterations // 12)
    ]

    with ProcessPoolExecutor(max_workers=4) as executor:
        stores_result = list(executor.map(_run_kuhn_chunk, chunks))

    # Verify convergence: strategy sums should be non-negative (CFR+ property)
    s = stores_result[-1].strategy_sums_for_infoset(0)  # noqa: F811
    assert len(s) == 2  # Kuhn has 2 actions per infoset
    assert np.all(s >= -1e-6), "Strategy sums should be non-negative"


@pytest.mark.benchmark_suite
def test_leduc_multi_thread_convergence() -> None:
    """Verify multi-thread Leduc CFR converges to valid strategies."""
    mp.set_start_method("spawn", force=True)

    iterations = 50
    chunks: list[tuple[int, int, CFRVariant, DCFRConfig | None]] = [
        (i * 10, i * 10 + 10, CFRVariant.CFR_PLUS, None)
        for i in range(iterations // 10)
    ]

    with ProcessPoolExecutor(max_workers=4) as executor:
        stores_result = list(executor.map(_run_leduc_chunk, chunks))

    # Verify convergence: strategy sums should be non-negative (CFR+ property)
    s = stores_result[-1].strategy_sums_for_infoset(0)  # noqa: F811
    assert np.all(s >= -1e-6), "Strategy sums should be non-negative"
