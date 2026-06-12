# PLAN.md

Purpose:
- Track the remaining work for the GPU poker solver.
- Keep the plan aligned with what already exists in the repo.
- Focus on the shortest path to a real runtime solver first.
- Keep the end goal visible: full abstracted 6-max NLHE with offline blueprint, runtime re-solving, and CUDA leaf eval.

How to use:
- `[ ]` not started
- `[-]` in progress
- `[x]` done

## 1. Already present

- [x] Card model and deck helpers
- [x] Board and street model
- [x] Betting, legality, rules, transitions, terminal, payouts
- [x] Game state and player state
- [x] Action model and baseline action abstraction
- [x] Public tree data structure
- [x] Toy-game CFR pieces for Kuhn and Leduc
- [x] Infoset storage and CFR iteration helpers
- [x] Leaf evaluator interface and CPU/GPU stubs
- [x] Runtime cache and warm-start scaffolding
- [x] Tests for core engine, tree, CFR, eval, and runtime pieces
- [x] Requirements include `pokerkit`, `treys`, `torch`, `numpy`, `scipy`, `numba`, `pytest`

## 2. Shortest MVP path

Goal:
- A heads-up postflop runtime solver that can take a real game state, build a depth-limited subtree, use the value network at the frontier, and return a strategy quickly.

Why this first:
- It proves the whole solve loop end to end.
- It keeps the tree small enough to debug.
- It creates the exact plumbing needed for larger 6-max abstractions later.

### 2.1 Runtime solve loop

- [x] Build public subtree from a live `GameState`
- [x] Support depth-limited leaves in the public tree
- [x] Evaluate frontier nodes through the leaf evaluator interface
- [x] Warm-start regrets and strategy sums from cache
- [x] Return root strategy and EV for a resolved spot
- [x] Wire the trained value network into `resolve_postflop_hu` as the default leaf evaluator
- [x] Add a runtime path that uses the value network for frontier EV instead of only the CPU stub
- [x] Add a deterministic fallback when the network is missing or mismatched
- [x] Add an end-to-end runtime test that solves one fixed postflop state

### 2.2 Value network integration

- [x] Dataset export pipeline exists
- [x] Value network training pipeline exists
- [x] Best checkpoint export exists
- [ ] Load `artifacts/value_run/best_checkpoint.pt` in runtime code
- [ ] Validate feature spec and label spec before inference
- [ ] Add a stable feature builder for runtime subtree leaves
- [ ] Make the leaf batch builder feed the trained network directly
- [ ] Compare network EV against CPU stub on a fixed validation spot
- [ ] Add a parity test for inference shape and output range

### 2.3 Solver correctness for MVP

- [ ] Add a tiny fixed postflop test tree with known actions
- [ ] Verify regret updates change strategy in the expected direction
- [ ] Verify caching changes nothing except speed
- [ ] Verify warm-start does not break determinism
- [ ] Verify card removal and range masking in the runtime path
- [ ] Verify canonical board handling is consistent between tree build and value labels

## 3. Near-term solver foundation

These are the pieces needed before the solver can grow beyond the MVP.

- [ ] Dense infoset storage for large abstract trees
- [ ] Flat array node representation for all solver paths
- [ ] Action masks per node and per position
- [ ] Regret matching and average strategy export
- [ ] CFR+ and DCFR variants in the main large-tree loop
- [ ] Time-budgeted iteration control for runtime solve calls
- [ ] Batched frontier evaluation for many leaves at once
- [ ] Subtree cache keyed by public state fingerprint

## 4. Full 6-max NLHE solver path

This is the end state. Every item here is part of the full production target.

### 4.1 Game abstraction

- [ ] Finalize separate preflop trees by position group
- [ ] Finalize postflop IP and OOP bet-size templates
- [ ] Add stack-to-pot-ratio aware action sets
- [ ] Add per-street action abstraction profiles
- [ ] Add suit-isomorphic board canonicalization shared by all modules
- [ ] Add blocker-aware card removal for every chance node
- [ ] Add private-hand indexing and range-vector normalization
- [ ] Add bucketed hand abstraction for postflop states

### 4.2 Multiway approximation

- [ ] Define the 6-max approximation model for 2-way and 3-way subgames
- [ ] Add coalition or aggregated-range handling for remaining players
- [ ] Add utilities for solving heads-up subgames inside larger hands
- [ ] Add a consistent policy for unresolved multiway branches
- [ ] Add tests that prove the approximation is stable across identical public states

### 4.3 Offline blueprint training

- [ ] Build the offline blueprint training loop on the abstracted tree
- [ ] Export blueprint regrets and average strategy tables
- [ ] Add checkpointing and resume for long runs
- [ ] Add pruning and stale-action removal for large abstract trees
- [ ] Add benchmark runs for convergence and throughput
- [ ] Add blueprint export format for runtime consumption

### 4.4 GPU CFR loop

- [ ] Move traversal-critical passes to dense tensor or block-sparse operations
- [ ] Batch reach-probability propagation on GPU
- [ ] Batch regret update on GPU
- [ ] Batch strategy accumulation on GPU
- [ ] Minimize Python overhead in the hot loop
- [ ] Add GPU/CPU parity tests for CFR math
- [ ] Add performance benchmarks for the large-tree loop

### 4.5 Runtime re-solving

- [ ] Build a live public-state resolver for 6-max spots
- [ ] Add subtree reconstruction from a current game state
- [ ] Add blueprint priors as warm starts
- [ ] Add depth-limit policies per street and SPR
- [ ] Add runtime leaf evaluation on the trained value network
- [ ] Add cache reuse across repeated public states
- [ ] Add latency budgets and graceful degradation
- [ ] Add a final action-selection API for external callers

### 4.6 End-to-end integration

- [ ] Export a blueprint from offline training
- [ ] Load that blueprint in runtime re-solving
- [ ] Resolve a live state using blueprint plus value network
- [ ] Emit root strategy, EV, and confidence metrics
- [ ] Add a CLI command for solve-from-state
- [ ] Add regression tests from saved states
- [ ] Add a reproducible benchmark suite for offline and runtime modes

## 5. Implementation order

Recommended order:

1. Wire the value network into depth-limited postflop solving.
2. Add a fixed-state end-to-end runtime test.
3. Add stronger feature and label validation for inference.
4. Generalize the runtime solver to more postflop spots.
5. Add blueprint export and reuse.
6. Expand abstraction and CFR to larger 6-max subgames.
7. Move the hot loop to GPU-backed dense or block-sparse execution.
8. Add full multiway approximation and production runtime plumbing.

## 6. Definition of done

The project is done when all of these are true:

- [ ] A real game state can be solved at runtime without manual plumbing.
- [ ] The solver uses the trained value network at depth limits.
- [ ] Offline blueprint training produces reusable strategy tables.
- [ ] GPU acceleration improves the largest bottleneck.
- [ ] Canonical board handling is shared and consistent everywhere.
- [ ] Multiway approximation is built into the solver path.
- [ ] End-to-end tests pass for offline and runtime modes.


