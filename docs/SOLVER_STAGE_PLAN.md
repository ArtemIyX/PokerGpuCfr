# Solver Stage Plan

This file tracks the implementation of the new solver-stage orchestration layer.

Goal:
- expose one entrypoint that runs a full CFR solve iteration
- keep the existing stage modules separate
- support CPU parallel branches, GPU leaf evaluation, and CFR-variant-specific regret updates

## Target pipeline

1. Stage 1: `ForwardProfile`
2. Stage 2: `AggregateProbSum`
3. Branch A on CPU:
   - Stage 3: `OpponentReach`
   - Stage 4: `ShowdownEquity`
4. Branch B on GPU:
   - Stage 5: `BatchLeafEval`
5. Join:
   - Stage 6: `BackwardCFV`
   - Stage 7: `UpdateRegret`

## High-level service API

Planned top-level request object:

- `game`: game variant, for now `kuhn` and `leduc`
- `cfr_variant`: `cfr`, `cfr_plus`, `dcfr`, `predictive_cfr_plus`
- `depth_limit`: integer depth cutoff for subtree solving
- `state`: exact state or sampled state spec
- `seed`: optional seed for random state generation
- `cpu_workers`: default parallelism for CPU stages
- `cpu_workers_stage3`: override for Stage 3
- `cpu_workers_stage4`: override for Stage 4
- `cpu_workers_stage6`: override for Stage 6
- `cpu_workers_stage7`: override for Stage 7
- `gpu_backend`: optional backend config for leaf evaluation
- `profiler`: optional profiler config
- `measure_timing`: whether to time each stage and branch
- `iterations`: number of solver iterations to run

Optional future request fields:

- `pruning_enabled`
- `abstraction_profile`
- `precision`
- `use_cudagraphs`
- `use_torch_compile`
- `checkpoint_path`
- `save_every`
- `exploitability_check`
- `return_intermediate_outputs`

## Implementation checklist

### Phase 1: API and data model

- [x] Define `GameVariant` enum or literal type
- [x] Define `CfrVariant` enum or literal type
- [x] Define `GameStateSpec`
- [x] Define `ProfilerSpec`
- [x] Define `TimingSpec`
- [x] Define `SolverStageRequest`
- [x] Define `SolverStageResult`
- [x] Add clear defaults for worker counts and timing flags

### Phase 2: Orchestration service

- [x] Add a solver-stage service module
- [x] Implement one public entrypoint for a full solve iteration
- [x] Wire Stage 1 and Stage 2 as the forward prefix
- [x] Split Stage 3/4 and Stage 5 into independent branches
- [x] Join branch outputs before Stage 6
- [x] Route CFR-variant selection only through Stage 7
- [x] Keep stage modules unchanged unless a bug requires a local fix

### Phase 3: Parallel execution

- [x] Use CPU workers for Stage 3
- [x] Use CPU workers for Stage 4
- [x] Use CPU workers for Stage 6
- [x] Use CPU workers for Stage 7
- [x] Keep GPU leaf evaluation isolated from CPU work
- [x] Make branch execution deterministic under fixed seed
- [x] Avoid leaking thread-pool details into stage modules

### Phase 4: Game support

- [x] Add Kuhn as the first minimal game target
- [x] Add Leduc as the next target
- [x] Add placeholder plumbing for future Kuhn / other variants if needed
- [x] Normalize game-specific tree construction behind one factory API
- [x] Make state generation work for exact and random input modes

### Phase 5: Profiling and timing

- [x] Add optional cProfile support
- [x] Add optional PyTorch profiler support for GPU work
- [x] Measure total iteration time
- [x] Measure per-stage time
- [x] Measure branch overlap time
- [x] Return timing data in the solve result

### Phase 6: Tests

- [x] Add unit tests for request validation
- [x] Add unit tests for branch orchestration
- [x] Add tests for CFR-variant routing in Stage 7
- [x] Add tests for game-variant selection
- [x] Add tests for exact vs random state setup
- [x] Add tests for timing metadata when enabled
- [x] Add tests for default worker behavior

### Phase 7: CLI / entrypoint integration

- [x] Add a command-line entrypoint for the solver stage
- [x] Accept game, variant, depth, and worker options from CLI
- [x] Accept optional seed and debug flags from CLI
- [x] Allow optional profiler flags
- [x] Print compact result summaries
- [x] Print detailed debug summaries with board, state, seed, and worker info
- [x] Save timing and diagnostic artifacts when requested

## Acceptance criteria

- one service call can run a full CFR iteration
- Stage 1 through Stage 7 remain separable and testable
- CPU and GPU branches are explicit in the orchestrator
- CFR variant changes only Stage 7 behavior
- Kuhn and Leduc can be selected through the same API
- timing can be enabled without changing solver logic

## Design rules

- keep hot-path logic in the stage modules
- keep the orchestration layer thin
- prefer small dataclasses over a huge flat argument list
- do not merge stage code just to simplify the service
- add tests before making structural changes that affect solver flow

## Open questions

- Should the first public API return only final outputs, or also intermediate stage artifacts?
- Should worker counts be one global default or one default per stage?
- Should profiling be recorded to memory, disk, or both?
- Should random state generation happen inside the service or in a separate game-state factory?
- Should Stage 5 accept a prebuilt batch or build its own batch from Stage 2 output?

## Root strategy variance bug

Observed issue:
- CLI debug output currently shows the same root strategy shape across seeds and setups
- seed is wired through the request, but the solver path does not yet expose a state-sensitive root evaluation in the CLI result

Fix steps:

### Step 1: Add regression tests

- [x] Add a unit test that compares root strategy output across two different seeds
- [x] Add a unit test that compares root strategy output across exact vs random state mode
- [x] Add a unit test that checks debug output includes seed, board, and state metadata
- [x] Add a larger solver test that varies board or state and asserts root strategy changes when the game state changes

### Step 2: Trace the state path

- [x] Add a solver test that verifies the request seed reaches the state-generation layer
- [x] Add a solver test that verifies the state spec affects the stage inputs used by Stage 1 / Stage 2
- [x] Add a solver test that verifies the printed root strategy is derived from the final solve state, not a fixed fallback

### Step 3: Fix the solver input path

- [ ] Add a concrete exact-state decoder or richer debug summary for the current game state
- [x] Thread the resolved game state into the solver-stage orchestration so the root evaluation is state-dependent
- [x] Recheck root strategy formatting after the solver receives a real state-dependent input
