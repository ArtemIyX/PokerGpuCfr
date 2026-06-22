# Hold'em HU Real Solver Plan

## Goal
Grow the current Hold'em heads-up solver from a tiny synthetic root spot into a real depth-limited HU Hold'em solver that:

- starts from an actual game state and street
- expands across multiple streets and betting rounds
- produces meaningful board-sensitive values
- carries multiple infosets and action branches
- remains small enough to test, debug, and iterate on

## Current State

The current Hold'em HU path is useful for plumbing validation, but it is still far too small:

- `tree_nodes=7`
- `tree_infosets=1`
- `tree_player_nodes=1`
- `tree_leaf_nodes=1`
- `tree_terminal_nodes=5`
- `tree_depth_limit_hits=1`

That means we are solving a toy decision with one root infoset and one truncated leaf, not a real poker subgame.

## Desired End State

The solver should support a realistic depth-limited Hold'em HU solve with:

- one root state per concrete game situation
- street-aware expansion: preflop, flop, turn, river
- multiple infosets per street or abstraction bucket
- branching by legal actions at each node
- leaf nodes only where depth truncation is intended
- board-sensitive evaluation that is informative but not numerically explosive
- deterministic behavior for a fixed seed/state

## Design Principles

- Keep the tree small enough to debug, but not so small that strategy is meaningless.
- Prefer explicit depth and node limits over hidden truncation.
- Keep leaf evaluation modest and board-sensitive.
- Preserve shape diagnostics so we can see what the solver is actually solving.
- Add tests before broadening the tree shape.

## Phase A: Define The Real HU Tree Shape

### Objectives

- Choose a target tree structure for a depth-limited HU Hold'em spot.
- Make the shape reflect actual betting progression instead of a single root spot.
- Keep the design compact enough to run quickly.

### Checklist

- [ ] Decide the minimum street coverage for the first real version.
- [ ] Decide how many decision nodes per street are required.
- [ ] Decide whether to model bet/call/fold/raise on each street or a reduced abstraction.
- [ ] Decide the leaf placement rules for truncation.
- [ ] Decide how much of the tree is exact versus abstracted.
- [ ] Document the target node/infoset count range.

### Recommended Target

- Preflop root decision
- Flop decision after one action path
- Turn decision after one action path
- River decision after one action path
- Leaf nodes only when the configured depth limit is reached
- Terminal nodes for fold/call/showdown endings

### Required Tests

- [ ] `tests/test_cfr_game_support.py`: Hold'em HU tree shape reflects the new multi-street design.
- [ ] `tests/test_holdem_hu_phase4.py`: root diagnostics show more than one infoset once the tree expands.
- [ ] `tests/test_holdem_hu_phase4.py`: depth-limit hit count is explicit and intentional.

### Exit Criteria

- The plan for the real tree shape is fixed and encoded in tests before implementation broadens the tree.

## Phase B: Expand Hold'em Tree Construction

### Objectives

- Replace the current toy root tree with a genuine multi-node Hold'em HU public tree.
- Generate nodes from actual legal actions and state transitions.
- Preserve node labels so debug output remains readable.

### Checklist

- [ ] Extend `src/pokergpu/cfr/solver/tree.py` to build more than one decision node.
- [ ] Reuse the generic tree builder when it produces a better state transition graph.
- [ ] Ensure each street advance is represented by distinct nodes.
- [ ] Ensure different actions do not collapse to the same child state.
- [ ] Keep action labels stable and informative.

### Implementation Notes

- The current root-only tree is not enough.
- The next version should create a chain or shallow tree across streets.
- A reduced action abstraction is acceptable if it preserves strategic differences.
- A tree with a few dozen nodes is still acceptable if it meaningfully separates streets.

### Required Tests

- [ ] `tests/test_holdem_hu_phase4.py`: tree node count increases beyond the current synthetic root spot.
- [ ] `tests/test_holdem_hu_phase4.py`: multiple infosets appear in the Hold'em tree.
- [ ] `tests/test_holdem_hu_phase4.py`: street transitions are visible in the public tree.
- [ ] `tests/test_cfr_solver_service.py`: tree diagnostics report the larger tree correctly.

### Bug Coverage

- Actions that all point to the same child.
- Streets that collapse into a single synthetic decision.
- Leaf nodes appearing too early.
- Terminal nodes being used where a leaf would be more appropriate.

### Exit Criteria

- Hold'em HU tree construction produces a multi-node, multi-street tree with meaningful action branching.

## Phase C: Make Leaf Evaluation Fit The Tree

### Objectives

- Keep leaf evaluation board-sensitive without overpowering the tree.
- Ensure heuristic leaf values are reasonable for depth-limited spots.
- Preserve the default model/triton backend for deeper or production-like use.

### Checklist

- [ ] Keep `HeuristicLeafBackend` bounded and board-sensitive.
- [ ] Ensure the default leaf backend remains the main backend.
- [ ] Make depth-limited behavior explicit in CLI and tests.
- [ ] Add stronger range checks for leaf outputs.

### Notes

- The heuristic backend should only be a fallback for depth-limited or debug spots.
- The triton/model backend should remain the main evaluation path where available.
- The evaluation scale should guide regrets, not dominate them.

### Required Tests

- [ ] `tests/test_holdem_hu_phase4.py`: heuristic leaf backend stays board-sensitive.
- [ ] `tests/test_holdem_hu_phase4.py`: heuristic leaf backend stays in a sane numeric range.
- [ ] `tests/test_holdem_hu_phase4.py`: default backend selection still uses the model/triton path.

### Exit Criteria

- Leaf evaluation remains informative but does not produce runaway values.

## Phase D: Validate CFR Behavior On The Larger Tree

### Objectives

- Confirm regrets and strategy sums move over repeated iterations on the expanded Hold'em tree.
- Confirm CFR variants remain stable on the larger shape.
- Confirm seed/state changes still produce deterministic outputs when they should.

### Checklist

- [ ] Verify repeated iterations continue to change root strategy.
- [ ] Verify root regrets accumulate in the expected direction.
- [ ] Verify CFR+, DCFR, and Predictive CFR+ still route correctly.
- [ ] Verify repeated run determinism for a fixed seed.

### Required Tests

- [ ] `tests/test_cfr_stage7.py`: repeated updates accumulate regrets correctly.
- [ ] `tests/test_holdem_hu_phase4.py`: root strategy changes over iterations on the repaired Hold'em spot.
- [ ] `tests/test_cfr_solver_variance.py`: different games and states still produce distinct strategies.

### Exit Criteria

- The larger tree still behaves like a CFR solver, not just a branching demo.

## Phase E: Expose Solver Work Statistics

### Objectives

- Make it obvious how much tree the solver is actually traversing.
- Track how much is terminal, leaf, and depth-limited.
- Help us compare toy spots against real-ish Hold'em spots.

### Checklist

- [x] Track total nodes.
- [x] Track player/chance/terminal/leaf nodes.
- [x] Track depth-limit hits.
- [x] Track infoset count.
- [x] Track branching statistics.
- [ ] Add street-specific counts if needed.
- [ ] Add maximum depth reached if useful.

### Required Tests

- [x] `tests/test_cfr_solver_service.py`: diagnostics include tree shape stats.
- [x] `tests/test_cfr_solver_service.py`: depth-limit hits are reported.

### Exit Criteria

- Every solver run tells us how much real tree is being solved and how much is being truncated.

## Phase F: Move Toward Real HU Poker

### Objectives

- Build toward a genuine heads-up Hold'em solver rather than a toy abstraction.
- Keep the system layered so we can still debug and benchmark small subgames.

### Checklist

- [ ] Add more realistic betting sequences.
- [ ] Expand from a single root branch to multiple street branches.
- [ ] Consider abstraction buckets or canonicalized states for larger branches.
- [ ] Keep pruning and depth-limiting explicit.
- [ ] Preserve the option to run tiny debug trees.

### Required Tests

- [ ] `tests/test_cfr_game_support.py`: new Hold'em tree remains deterministic.
- [ ] `tests/test_holdem_hu_phase4.py`: diagnostics show the tree is larger than the toy root spot.
- [ ] `tests/test_solver_holdem_hu_cli.py`: CLI output still reports the exact solved state.

### Exit Criteria

- The Hold'em HU solver is solving a real structured subgame, not just a single-step toy model.

## Recommended Order

1. Phase A, define the target shape.
2. Phase B, expand the public tree.
3. Phase C, keep leaf evaluation sane.
4. Phase D, validate CFR dynamics on the larger tree.
5. Phase E, keep work statistics visible.
6. Phase F, continue moving toward a real HU solver.

## Success Definition

We will know the solver is heading toward a real Hold'em HU system when:

- the tree contains multiple infosets across streets
- the depth-limit hit count is intentional and not the whole tree
- the solver can distinguish boards and streets meaningfully
- repeated CFR updates still accumulate strategy and regret correctly
- the diagnostics show a non-toy amount of work

