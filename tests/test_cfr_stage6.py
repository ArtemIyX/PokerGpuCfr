from __future__ import annotations

import numpy as np
import pytest

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityBatchInput
from pokergpu.cfr.stage4 import ShowdownEquityResult
from pokergpu.cfr.stage6 import BackwardCFVInput
from pokergpu.cfr.stage6 import backward_cfv
from pokergpu.cfr.stage7 import update_regret
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def test_backward_cfv_combines_leaf_showdown_and_infoset_values() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.PLAYER0,
            NodeType.LEAF,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1)),
            ChildLink(child=NodeId(2)),
        ),
        infoset_ids=(InfosetId(0), None, None),
        terminal_payoffs=(None, None, Chips(2)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.5, 0.5),
        infoset_reach=(1.0,),
        action_reach=((0.25, 0.75), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)
    showdown = ShowdownEquityResult(
        node_showdown_equity=(0.0, 0.0, 2.0),
        node_showdown_equity_bb=(0.0, 0.0, 2.0),
        input_rows=ShowdownEquityBatchInput(rows=()),
        output_rows=(),
    )
    stage6_input = BackwardCFVInput(
        tree=tree,
        forward=forward,
        aggregate=aggregate,
        opponent_reach=opponent,
        showdown=showdown,
        leaf_values=np.asarray((1.5,), dtype=np.float64),
    )

    result = backward_cfv(stage6_input)

    assert result.node_values[1] == 1.5
    assert result.node_values[2] == 2.0
    assert result.node_values[0] == 1.875
    assert result.infoset_values[0] == 1.875
    assert result.action_values[0] == (1.5, 2.0)


def test_backward_cfv_propagates_through_chance_nodes() -> None:
    tree = PublicTree(
        node_types=(
            NodeType.CHANCE,
            NodeType.TERMINAL,
            NodeType.TERMINAL,
        ),
        first_child=(0, 2, 2),
        child_count=(2, 0, 0),
        children=(
            ChildLink(child=NodeId(1), chance_prob=0.25),
            ChildLink(child=NodeId(2), chance_prob=0.75),
        ),
        infoset_ids=(None, None, None),
        terminal_payoffs=(None, Chips(4), Chips(0)),
    )
    forward = ForwardProfileResult(
        node_reach=(1.0, 0.25, 0.75),
        infoset_reach=(),
        action_reach=((), (), ()),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)
    showdown = ShowdownEquityResult(
        node_showdown_equity=(0.0, 4.0, 0.0),
        node_showdown_equity_bb=(0.0, 4.0, 0.0),
        input_rows=ShowdownEquityBatchInput(rows=()),
        output_rows=(),
    )
    stage6_input = BackwardCFVInput(
        tree=tree,
        forward=forward,
        aggregate=aggregate,
        opponent_reach=opponent,
        showdown=showdown,
        leaf_values=np.asarray((), dtype=np.float64),
    )

    result = backward_cfv(stage6_input)

    assert result.node_values[0] == pytest.approx(1.0)
    assert result.node_values[1] == pytest.approx(4.0)
    assert result.node_values[2] == pytest.approx(0.0)
    assert result.action_values[0] == (4.0, 0.0)


def test_update_regret_still_matches_stage7_contract() -> None:
    assert update_regret((1.0, 2.0), (3.0, 4.0), 2.5) == (1.5, 3.5)
