# Hold'em HU Strategy Repair Plan

## Goal
Make the Hold'em heads-up solver produce non-uniform, state-sensitive strategies that actually reflect regret updates instead of falling back to the default uniform distribution.

## Problem Summary
The current CLI prints a stable-looking `root_strategy` across different seeds because the underlying solve path is too compressed:

- The Hold'em HU public tree is a fixed 4-node street skeleton.
- The CLI often solves with an empty or under-specified board.
- Leaf evaluation can be too symmetric or constant in practice.
- If regrets stay non-positive, strategy reconstruction falls back to uniform.

The repair work should separate these causes and fix them in order.

## Phase 1: Verify The Failure Mode

### Objectives

- Confirm whether the solver is producing near-zero regrets, or whether regret updates are being computed but discarded later.
- Confirm whether seed changes affect the actual solved state, or only the printed wrapper state.
- Confirm whether the current tree structure is too shallow to express meaningful strategy differences.

### Checklist

- [x] Print root regret sums alongside root strategy for at least one run.
- [x] Print root action values and node value for the same run.
- [x] Compare runs with different seeds and identical board/state settings.
- [x] Compare runs with explicit board inputs versus empty-board inputs.
- [x] Confirm whether the root strategy is identical because regrets are identical, or because the strategy formatter is defaulting to uniform.

### Required Tests

- [x] `tests/test_solver_holdem_hu_cli.py`: root strategy should not be derived from an empty/default fallback when regrets are non-zero.
- [x] `tests/test_cfr_solver_service.py`: root diagnostics are reported by the solver stage.
- [x] `tests/test_holdem_hu_phase4.py`: Hold'em HU CLI prints the new root diagnostics in debug mode.
- [x] `tests/test_cfr_stage6.py`: action values must differ when child values differ.

### Bug Coverage

- Root strategy always uniform because regrets are zero.
- Root strategy always uniform because strategy formatting ignores updated regrets.
- Seed changes do not affect the actual game state being solved.
- Leaf values are constant enough that regrets never separate.

### Exit Criteria

- We can explain, with numbers, whether the uniform strategy comes from tree collapse, weak evaluation, or update logic.

Status: complete. The root diagnostics were added and used to identify the failure as tree/value collapse rather than strategy extraction.

## Phase 2: Make The Solved State Real

### Objectives

- Ensure the solver is solving a concrete Hold'em state, not just a generic variant label.
- Make board and betting context part of the actual solve input.
- Prevent hidden defaults from silently collapsing the problem into a preflop-like placeholder.

### Checklist

- [x] Review `src/pokergpu/solver_holdem_hu_cli.py` state construction.
- [x] Replace or augment blank-board defaults with explicit board/state resolution.
- [x] Ensure the CLI refuses to pretend an unspecified state is a fully specified spot.
- [x] Ensure seed-based random state generation is visible in summary output.

### Required Tests

- [x] `tests/test_solver_holdem_hu_cli.py`: explicit board input changes the reported state and downstream root strategy.
- [x] `tests/test_solver_holdem_hu_cli.py`: identical inputs produce identical strategies.
- [x] `tests/test_solver_holdem_hu_cli.py`: different seeded states can produce different legal actions or different root strategies.

### Bug Coverage

- CLI says it solved a spot, but the tree input was effectively blank.
- Different seeds only change a dummy wrapper state, not the solver state.
- State serialization/deserialization loses board or betting information.

### Exit Criteria

- The solver summary shows the exact state being solved, and that state is actually used by the tree/evaluation path.

## Phase 3: Expand Or Fix The Public Tree

### Objectives

- Replace the overly shallow street skeleton with a depth-limited game tree that reflects actual betting structure.
- Ensure decision nodes are generated from legal actions in the current state.
- Ensure different actions really lead to different child states and eventually different leaf values.

### Checklist

- [x] Audit `src/pokergpu/cfr/solver/tree.py`.
- [x] Prefer the generic tree builder for Hold'em HU if it preserves actual action/state transitions.
- [x] Ensure the tree contains enough decision points to express distinct lines.
- [x] Ensure terminal and leaf nodes are placed by game rules, not just by street index.
- [x] Preserve action labels so printed strategy is interpretable.

### Required Tests

- [x] `tests/test_holdem_hu_phase4.py`: Hold'em HU root action values are not all zero after tree repair.
- [x] `tests/test_holdem_hu_phase4.py`: the Hold'em HU tree now exposes distinct root children.
- [x] `tests/test_tree_builder.py`: terminal/leaf classification must match the actual game phase.

### Bug Coverage

- A fixed shallow tree makes all strategy outputs look similar.
- Actions with different EVs collapse into the same child path.
- Leaf nodes appear too early, before strategy can separate.
- Action labels become misaligned with the number of children.

### Exit Criteria

- The tree contains enough structure for CFR to differentiate betting lines, folds, calls, and raises.

Status: partially complete. The specific root-collapse bug is fixed, and the root now has distinct action values. The broader goal of expanding the Hold'em HU tree into a more realistic street-by-street betting tree is still pending.

## Phase 4: Make Leaf Evaluation Informative

### Objectives

- Ensure leaf values vary with board texture, hand strength, and action line.
- Prevent the solver from feeding nearly constant values into regret matching.
- Make sure postflop and truncated-depth spots do not collapse into symmetric heuristics.

### Checklist

- [x] Audit `src/pokergpu/cfr/solver/evaluation.py`.
- [x] Verify the `board is None` path does not accidentally zero out the useful part of evaluation.
- [x] Verify heuristic leaf backend output changes across materially different states.
- [x] Verify showdown equity is board-aware and not constant across all actions.
- [x] Confirm leaf batches carry enough features to distinguish different states.

### Required Tests

- [x] `tests/test_cfr_stage4.py`: showdown equity should differ between clearly stronger and weaker hands.
- [x] `tests/test_cfr_stage2.py`: leaf feature rows should change when the board changes.
- [x] `tests/test_leaf_backend.py`: heuristic/model/triton backends should not all return identical outputs for distinct inputs.
- [x] `tests/test_cfr_stage6.py`: backward CFV should propagate different leaf values into different action values.

### Bug Coverage

- Constant leaf values produce equal regrets and uniform strategy.
- Boardless or placeholder evaluation causes all actions to look equally good.
- Feature construction does not encode enough state to inform a value model.

### Exit Criteria

- At least one realistic spot produces visibly different leaf values for different actions or lines.

Status: complete. The Hold'em leaf path is now board-sensitive and is covered by direct and end-to-end tests.

## Phase 5: Fix Regret And Strategy Update Behavior

### Objectives

- Make sure regret updates are numerically sound and not accidentally overwritten.
- Verify the solver accumulates average strategy from the current policy, not from a fallback path.
- Ensure CFR variant logic does not erase meaningful regret structure.

### Checklist

- [ ] Audit `src/pokergpu/cfr/stage7.py` and `src/pokergpu/cfr/solver/strategy_update.py`.
- [ ] Confirm positive-regret accumulation matches CFR expectations.
- [ ] Confirm CFR+ and DCFR transformations do not collapse all regrets to the same pattern.
- [ ] Check that strategy sums are accumulated with the correct reach weights.
- [ ] Check that the strategy formatter uses the intended source table.

### Required Tests

- [ ] `tests/test_cfr_iteration.py`: positive action values should produce non-uniform regret updates.
- [ ] `tests/test_cfr_iteration.py`: strategy sums should change after updates and retain asymmetry.
- [ ] `tests/test_cfr_iteration.py`: CFR+, DCFR, and baseline CFR should preserve expected variant-specific behavior.
- [ ] `tests/test_cfr_stage7.py`: regret matching should return a non-uniform policy when positive regrets differ.

### Bug Coverage

- Regrets are updated but strategy reconstruction still falls back to uniform.
- Strategy sums are accumulated with the wrong reach weights.
- CFR+ clamping or DCFR discounting erases all asymmetry.
- Updated regrets are not persisted across iterations.

### Exit Criteria

- A simple asymmetric toy state produces a visibly asymmetric policy after updates.

Status: next up. The current focus is to verify that repeated iterations keep moving root strategy and regrets in the expected direction on the repaired Hold'em spot.

## Phase 6: Add End-To-End Solver Regression Coverage

### Objectives

- Prove the solver reacts to different spots, not just different random seeds.
- Make “uniform every time” a hard regression.
- Lock in the expected behavior for CLI output and internal solver state.

### Checklist

- [ ] Add a small end-to-end test around `src/pokergpu/solver_holdem_hu_cli.py`.
- [ ] Run at least one preflop and one postflop spot.
- [ ] Run at least one spot with intentionally skewed leaf values.
- [ ] Confirm root strategy is not exactly uniform in a solved asymmetric case.
- [ ] Confirm repeated runs with the same input are deterministic.

### Required Tests

- [ ] `tests/test_solver_holdem_hu_cli.py`: asymmetric spot yields non-uniform root strategy.
- [ ] `tests/test_solver_holdem_hu_cli.py`: identical input and seed yields identical strategy.
- [ ] `tests/test_solver_holdem_hu_cli.py`: different spot or board yields different strategy.
- [ ] `tests/test_solver_holdem_hu_cli.py`: summary output includes root strategy, state, board, and legal actions.

### Bug Coverage

- Silent regressions where the solver still prints plausible-looking uniform results.
- Nondeterminism when the same seed/state should be reproducible.
- CLI summary drift that hides the actual solved state.

### Exit Criteria

- At least one integration test fails if the solver regresses to a uniform fallback policy.

## Phase 7: Performance Sanity Check

### Objectives

- Keep the fix from making the solver materially slower.
- Ensure any added diagnostics stay out of hot paths.
- Preserve the current parallel structure where it matters.

### Checklist

- [ ] Verify added prints/logging are debug-only.
- [ ] Ensure any new tree expansion remains bounded by depth and node limits.
- [ ] Ensure any additional tests use tiny trees/spots.
- [ ] Check that no new Python loops land in the hot evaluation path without reason.

### Required Tests

- [ ] `tests/test_performance_smoke.py`: small benchmark does not regress beyond an agreed threshold.
- [ ] `tests/test_solver_holdem_hu_cli.py`: debug output can be disabled.

### Bug Coverage

- Fixing correctness by adding expensive per-node Python work.
- Debug instrumentation leaking into normal runs.

### Exit Criteria

- Correctness improves without a visible regression in the intended runtime path.

## Recommended Implementation Order

1. Phase 1, to identify the dominant failure mode.
2. Phase 3 root-tree repair, to ensure the root has distinct action values.
3. Phase 2, to ensure the solver is working on a real spot.
4. Phase 4, to ensure leaf values can actually drive regrets.
5. Phase 5, to validate regret accumulation and policy extraction.
6. Phase 6, to lock the behavior with integration tests.
7. Phase 7, to confirm the fix stays practical.

## Success Definition

The solver is considered repaired when all of the following are true:

- Different game states can produce different root strategies.
- Asymmetric toy spots produce non-uniform policies after solving.
- Re-running the same input with the same seed is deterministic.
- The printed root strategy matches the internal regret state.
- The solver no longer falls back to uniform because the entire pipeline is numerically flat.
