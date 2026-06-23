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

- [x] Decide the minimum street coverage for the first real version.
- [x] Decide how many decision nodes per street are required.
- [x] Decide whether to model bet/call/fold/raise on each street or a reduced abstraction.
- [x] Decide the leaf placement rules for truncation.
- [x] Decide how much of the tree is exact versus abstracted.
- [x] Document the target node/infoset count range.

### Recommended Target

- Minimum street coverage for version 1:
  - preflop, flop, turn, river
- Decision nodes per street:
  - one player decision node per street on the main line
  - optional opponent response nodes only where they add real strategic separation
- Action abstraction:
  - `check`
  - `bet:25pct`
  - `bet:50pct`
  - `bet:75pct`
  - `bet:100pct`
  - `bet:150pct`
  - plus `call`, `fold`, and `raise` only when the spot is structurally relevant
- Leaf placement:
  - create leaf nodes when `depth_limit` is reached before showdown
  - create terminal nodes only for true fold / all-in / showdown endings
- Exact vs abstracted:
  - exact board and betting state transitions on the main line
  - abstracted action set and depth-limited subtree expansion
  - no attempt yet to enumerate the full no-limit action space
- Target size:
  - not a full solver tree yet
  - roughly `20-80` nodes for the first real version
  - at least `4` infosets, ideally `8+` once the branching is in place

### Required Tests

- [ ] `tests/test_cfr_game_support.py`: Hold'em HU tree shape reflects the new multi-street design.
- [ ] `tests/test_holdem_hu_phase4.py`: root diagnostics show more than one infoset once the tree expands.
- [ ] `tests/test_holdem_hu_phase4.py`: depth-limit hit count is explicit and intentional.

### Exit Criteria

- The plan for the real tree shape is fixed and encoded in tests before implementation broadens the tree.

Status: decided. We are targeting a compact but real four-street HU tree with abstracted actions and explicit depth-limited leaves.

## Phase B: Expand Hold'em Tree Construction

### Objectives

- Replace the current toy root tree with a genuine multi-node Hold'em HU public tree.
- Generate nodes from actual legal actions and state transitions.
- Preserve node labels so debug output remains readable.

### Checklist

- [x] Extend `src/pokergpu/cfr/solver/tree.py` to build more than one decision node.
- [x] Reuse the generic tree builder when it produces a better state transition graph.
- [x] Ensure each street advance is represented by distinct nodes.
- [x] Ensure different actions do not collapse to the same child state.
- [x] Keep action labels stable and informative.

### Implementation Notes

- The current root-only tree is not enough.
- The next version should create a chain or shallow tree across streets.
- A reduced action abstraction is acceptable if it preserves strategic differences.
- A tree with a few dozen nodes is still acceptable if it meaningfully separates streets.

### Required Tests

- [x] `tests/test_holdem_hu_phase4.py`: tree node count increases beyond the current synthetic root spot.
- [x] `tests/test_holdem_hu_phase4.py`: multiple infosets appear in the Hold'em tree.
- [x] `tests/test_holdem_hu_phase4.py`: street transitions are visible in the public tree.
- [x] `tests/test_cfr_solver_service.py`: tree diagnostics report the larger tree correctly.

### Bug Coverage

- Actions that all point to the same child.
- Streets that collapse into a single synthetic decision.
- Leaf nodes appearing too early.
- Terminal nodes being used where a leaf would be more appropriate.

### Exit Criteria

- Hold'em HU tree construction produces a multi-node, multi-street tree with meaningful action branching.

Status: implemented as a first-pass compact multi-street chain. The tree is no longer a single root toy spot, but it still needs refinement toward a richer poker subgame.

## Phase C: Make Leaf Evaluation Fit The Tree

### Objectives

- Keep leaf evaluation board-sensitive without overpowering the tree.
- Ensure heuristic leaf values are reasonable for depth-limited spots.
- Preserve the default model/triton backend for deeper or production-like use.

### Checklist

- [x] Keep `HeuristicLeafBackend` bounded and board-sensitive.
- [x] Ensure the default leaf backend remains the main backend.
- [x] Make depth-limited behavior explicit in CLI and tests.
- [x] Add stronger range checks for leaf outputs.

### Notes

- The heuristic backend should only be a fallback for depth-limited or debug spots.
- The triton/model backend should remain the main evaluation path where available.
- The evaluation scale should guide regrets, not dominate them.

### Required Tests

- [x] `tests/test_holdem_hu_phase4.py`: heuristic leaf backend stays board-sensitive.
- [x] `tests/test_holdem_hu_phase4.py`: heuristic leaf backend stays in a sane numeric range.
- [x] `tests/test_holdem_hu_phase4.py`: default backend selection still uses the model/triton path.

### Exit Criteria

- Leaf evaluation remains informative but does not produce runaway values.

Status: implemented. The heuristic fallback is now bounded and board-sensitive, the default solver path uses the model/Triton backend, and leaf outputs are validated for finite values and sane ranges.

## Phase D: Validate CFR Behavior On The Larger Tree

### Objectives

- Confirm regrets and strategy sums move over repeated iterations on the expanded Hold'em tree.
- Confirm CFR variants remain stable on the larger shape.
- Confirm seed/state changes still produce deterministic outputs when they should.

### Checklist

- [x] Verify repeated iterations continue to change root strategy.
- [x] Verify root regrets accumulate in the expected direction.
- [x] Verify CFR+, DCFR, and Predictive CFR+ still route correctly.
- [x] Verify repeated run determinism for a fixed seed.

### Required Tests

- [x] `tests/test_cfr_stage7.py`: repeated updates accumulate regrets correctly.
- [x] `tests/test_holdem_hu_phase4.py`: root strategy changes over iterations on the repaired Hold'em spot.
- [x] `tests/test_cfr_solver_variance.py`: different games and states still produce distinct strategies.

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

### Required Tests

- [x] `tests/test_cfr_solver_service.py`: diagnostics include tree shape stats.
- [x] `tests/test_cfr_solver_service.py`: depth-limit hits are reported.

### Exit Criteria

- Every solver run tells us how much real tree is being solved and how much is being truncated.

## Phase F: Move Toward Real HU Poker

### Objectives

- Build toward a genuine heads-up Hold'em solver rather than a toy abstraction.
- Make the public tree reflect realistic betting progression across streets.
- Keep the system layered so we can still debug and benchmark small subgames.
- Preserve deterministic, depth-limited behavior while the tree becomes richer.
- Keep the default solver path practical for development and debugging.

### Checklist

- [x] Add street-specific betting sequences with distinct preflop, flop, turn, and river action sets.
- [x] Expand from a single root branch to multiple street branches with meaningful public state separation.
- [x] Ensure fold, call, check, bet, and raise transitions remain distinct when they should be.
- [x] Keep pruning and depth-limiting explicit in tree construction and CLI wiring.
- [x] Preserve the option to run tiny debug trees with a reduced abstraction profile.
- [ ] Consider abstraction buckets or canonicalized states only where they reduce tree size without collapsing strategic differences.
- [x] Keep board-sensitive leaf evaluation aligned with the larger tree shape.
- [x] Keep repeated CFR runs deterministic for a fixed seed or exact state.
- [x] Keep diagnostics readable enough to inspect tree growth, street coverage, and truncation.

### Progress Notes

- Added street-specific Hold'em HU bet sizing in `src/pokergpu/abstraction/actions.py`.
- Expanded the Hold'em public tree scaffold to preserve distinct street-labeled branches in `src/pokergpu/cfr/solver/tree.py`.
- Updated tree-shape tests to expect a richer multi-street scaffold in `tests/test_cfr_game_support.py`.
- Added a transition regression test in `tests/test_transitions.py` to ensure fold, call, and raise lead to distinct next states when legal.
- Added an explicit compact-tree flag in `src/pokergpu/solver_holdem_hu_cli.py` so tiny debug trees can be requested directly.
- Added a compact Hold'em tree mode in `src/pokergpu/cfr/solver/tree.py` for reduced-abstraction debug runs.
- Threaded `depth_limit` through the tree factory and Hold'em CLIs so shallow solves can select a compact tree path explicitly.
- Added tree-shape diagnostics in `src/pokergpu/cfr/solver/service.py` for node counts, branching, and action-label variation.
- Added a Hold'em leaf-eval regression test that checks distinct values across preflop, flop, turn, and river boards.

### Required Tests

- [x] `tests/test_cfr_game_support.py`: new Hold'em tree remains deterministic.
- [x] `tests/test_holdem_hu_phase4.py`: diagnostics show the tree is larger than the toy root spot.
- [ ] `tests/test_holdem_hu_phase4.py`: street transitions and action labels remain distinct across the expanded branches.
- [ ] `tests/test_holdem_hu_phase4.py`: repeated runs with the same seed or exact state remain deterministic.
- [ ] `tests/test_solver_holdem_hu_cli.py`: CLI output still reports the exact solved state.
- [ ] `tests/test_cfr_solver_variance.py`: different boards and seeds continue to produce distinct strategies.
- [ ] `tests/test_cfr_stage7.py`: the expanded tree still accepts all supported CFR variants.

### Exit Criteria

- The Hold'em HU solver is solving a real structured subgame, not just a single-step toy model.
- The public tree has multiple street branches and the solver remains deterministic and debuggable.
- The solver still supports the small debug configuration without losing the realistic path.

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
