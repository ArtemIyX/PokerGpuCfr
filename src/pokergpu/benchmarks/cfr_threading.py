"""Parallel CFR benchmark utilities for testing single vs multi-thread performance."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from pokergpu.cfr.infosets import InfosetStore
from pokergpu.cfr.iteration import CFRVariant, DCFRConfig, run_cfr_iteration
from pokergpu.cfr.kuhn import (
    expected_action_utilities as kuhn_expected_action_utilities,
    kuhn_infoset_indices_for_player,
    new_kuhn_infoset_store,
)
from pokergpu.cfr.leduc import (
    expected_action_utilities_leduc,
    leduc_infoset_indices_for_player,
    new_leduc_infoset_store,
)


def _train_kuhn_parallel(
    iter_range: tuple[int, int],
    player_infoset_indices: list[int],
    variant: CFRVariant = CFRVariant.CFR_PLUS,
    dcfr_config: DCFRConfig | None = None,
) -> InfosetStore:  # noqa: F811
    """Train Kuhn CFR in parallel for a range of iterations.

    Args:
        iter_range: Tuple of (start_iteration, end_iteration) inclusive.
        player_infoset_indices: All infoset indices for both players.
        variant: CFR variant to use.
        dcfr_config: DCFR configuration if using DCFR.

    Returns:
        Trained infoset store.
    """
    store = new_kuhn_infoset_store()
    iteration_end = iter_range[1]

    for it in range(iter_range[0], iteration_end + 1):
        utilities = kuhn_expected_action_utilities(store, 0)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=player_infoset_indices[:3],
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )
        utilities = kuhn_expected_action_utilities(store, 1)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=player_infoset_indices[3:],
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )

    return store


def _train_leduc_parallel(
    iter_range: tuple[int, int],
    player_infoset_indices: list[int],
    variant: CFRVariant = CFRVariant.CFR_PLUS,
    dcfr_config: DCFRConfig | None = None,
) -> InfosetStore:  # noqa: F811
    """Train Leduc CFR in parallel for a range of iterations.

    Args:
        iter_range: Tuple of (start_iteration, end_iteration) inclusive.
        player_infoset_indices: All infoset indices for both players.
        variant: CFR variant to use.
        dcfr_config: DCFR configuration if using DCFR.

    Returns:
        Trained infoset store.
    """
    store = new_leduc_infoset_store()
    iteration_end = iter_range[1]

    for it in range(iter_range[0], iteration_end + 1):
        utilities = expected_action_utilities_leduc(store, 0)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=player_infoset_indices[:5],
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )
        utilities = expected_action_utilities_leduc(store, 1)
        run_cfr_iteration(
            store=store,
            action_utilities=utilities,
            active_infosets=player_infoset_indices[5:],
            variant=CFRVariant.CFR_PLUS,
            iteration=it,
            dcfr_config=dcfr_config,
        )

    return store
