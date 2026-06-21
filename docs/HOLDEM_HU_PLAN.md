# Hold'em HU Implementation Plan

This document is the working roadmap for adding heads-up no-limit Texas Hold'em to the existing CFR solver stack.

The goal is not full exact NLHE solving. The goal is a practical, depth-limited, GPU-friendly heads-up path that reuses the current CFR pipeline and scales step by step:

- keep the current Kuhn and Leduc solver paths working;
- add a Hold'em HU public tree;
- add street-aware action abstraction;
- add preflop and postflop card abstraction;
- keep depth-limited leaf evaluation as the default;
- preserve the flat-array / batched evaluation design.

## Scope

In scope:

- heads-up only, 1 vs 1;
- no-limit betting abstraction with a fixed global layout;
- depth-limited solving;
- card buckets for preflop and postflop;
- CPU correctness first, GPU integration second;
- regression tests for every new layer.

Out of scope for this phase:

- full exact NLHE tree construction;
- 6-max support;
- perfect-information exact solve of all streets;
- advanced pruning or abstraction learning beyond the initial bucket scheme.

## Design Decisions

These are the main decisions we should keep consistent while implementing.

### 1. Use a street-aware global action abstraction

Why:

- preflop and postflop behave differently;
- pot-relative bet sizing is natural postflop but awkward preflop;
- a single action template is easier to test and reason about than per-node ad hoc sizing.

Decision:

- keep one global abstraction profile;
- split it by street; (some streets has smaller layout, some streets larger)
- use a shared action vocabulary across the tree:
  - `fold`
  - `check`
  - `call`
  - `bet 25% pot`
  - `bet 50% pot`
  - `bet 75% pot`
  - `bet 100% pot`
  - `bet 150% pot`
  - `all-in`

Important note:

- preflop should not blindly reuse postflop bet percentages;
- preflop should either use a separate smaller template or map the same vocabulary through a preflop-specific sizing rule.

### 2. Keep depth-limited solving as the default Hold'em path

Why:

- the current runtime already assumes leaf evaluation and public-tree truncation;
- depth-limited solving is the right bridge to a value network;
- it keeps the first HU implementation tractable and testable.

Decision:

- treat a chosen depth or street cutoff as the public-tree frontier;
- evaluate frontier leaves with heuristic or learned leaf evaluation;
- do not attempt a complete exact hand-game tree first.

### 3. Use separate preflop and postflop card abstraction

Why:

- postflop hand strength depends on the board;
- preflop bucketing should reflect hole-card value without board context;
- a single bucket scheme would be too coarse for both regimes.

Decision:

- preflop: hand-class or equity-based buckets;
- postflop: reuse the existing strength-tier bucketer;
- keep bucket outputs flat and fixed-size so they can feed dense CFR or a value network.

### 4. Preserve the current CFR pipeline instead of replacing it

Why:

- the current solver already has a staged architecture;
- we want Hold'em to plug into the same solver flow;
- changing the pipeline and the game model at the same time would make debugging much harder.

Decision:

- extend the tree and abstraction layers first;
- leave Stage 1 to Stage 7 interfaces stable where possible;
- only introduce new data where Hold'em requires it.

## Files To Edit

These are the likely files for the first implementation pass.

### Core model and tree

- [`src/pokergpu/cfr/solver/tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py)
- [`src/pokergpu/cfr/solver/spec.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/spec.py)
- [`src/pokergpu/solver_cli.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/solver_cli.py)

### Action abstraction

- [`src/pokergpu/abstraction/actions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/actions.py)
- [`tests/test_action_abstraction.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/tests/test_action_abstraction.py)

### Card abstraction

- [`src/pokergpu/abstraction/buckets.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/buckets.py)
- [`src/pokergpu/abstraction/hands.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/hands.py)
- [`tests/test_buckets.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/tests/test_buckets.py)
- [`tests/test_hand_ranges.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/tests/test_hand_ranges.py)

### Solver integration

- [`src/pokergpu/cfr/solver/__init__.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/__init__.py)
- [`src/pokergpu/cfr/solver/leduc_solver.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/leduc_solver.py)
- [`src/pokergpu/cfr/solver/kuhn_solver.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/kuhn_solver.py)
- new Hold'em-specific solver module, if needed

### Runtime and tests

- [`tests/test_cfr_game_support.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/tests/test_cfr_game_support.py)
- new Hold'em tree tests
- new depth-limit / leaf-evaluation tests
- new solver smoke tests

## Phase 0 - Confirm the Hold'em data model

Goal:

- define what a Hold'em HU public state needs to represent before building the tree.

Checklist:

- [ ] confirm blind structure and stack model for heads-up no-limit;
- [ ] confirm how dealer / button position is encoded;
- [ ] confirm how many community cards are available at each street;
- [ ] confirm whether the tree starts from preflop action or from an arbitrary public state;
- [ ] confirm how leaf cutoff is expressed: by depth, by street, or both.

Expected outputs:

- a stable public-state shape for HU Hold'em;
- a small decision about whether the tree builder takes a full `GameState` or a reduced Hold'em state object.

Tests to add or update after Phase 0:

- state-construction test for a minimal HU Hold'em public state;
- validation test for impossible Hold'em states;
- regression test that Kuhn and Leduc state handling still works.

## Phase 1 - Add Hold'em HU tree scaffolding

Goal:

- make `GameVariant.HOLDEM_HU` build a public tree instead of raising `NotImplementedError`.

Checklist:

- [ ] add `make_holdem_hu_public_tree`;
- [ ] wire `GameVariant.HOLDEM_HU` into the tree factory;
- [ ] keep the tree shallow at first with a configurable depth limit;
- [ ] preserve the flat public-tree representation;
- [ ] keep terminal and leaf node classification explicit.

Key decision:

- start with a compact abstraction tree, not the full game tree.

Why:

- the full Hold'em tree is too large to validate early;
- a shallow tree lets us verify action generation and depth-limited cutoffs quickly.

Tests to add or update after Phase 1:

- tree factory returns a non-empty Hold'em tree;
- tree factory still rejects unsupported variants only where intended;
- root node type and child counts are consistent with the starting street;
- depth-limit cutoff produces leaf nodes where expected;
- existing Kuhn and Leduc tree tests still pass.

## Phase 2 - Implement street-aware action abstraction

Goal:

- make Hold'em betting sizes explicit and deterministic.

Checklist:

- [ ] extend the action abstraction profile to encode street-specific templates;
- [ ] define the global action layout;
- [ ] encode separate preflop and postflop sizing rules;
- [ ] ensure bet and raise amounts clamp correctly to stack size and minimum legal sizing;
- [ ] prevent duplicate actions after rounding/clamping;
- [ ] keep the abstraction output deterministic for tests.

Key decision:

- use pot-relative postflop bet sizing, but do not copy that logic blindly into preflop.

Why:

- postflop pot-relative sizes are natural and easy to reason about;
- preflop often needs a different policy because pot size is not yet a good proxy for action pressure.

Tests to add or update after Phase 2:

- preflop action enumeration includes the intended legal subset;
- postflop action enumeration includes the intended percentage sizes;
- facing-bet states return fold/call/raise when legal;
- unopened-pot states return check/bet when legal;
- clamping to stack and minimum raise behaves correctly;
- no duplicate actions are emitted.

## Phase 3 - Add preflop and postflop card buckets

Goal:

- reduce 1326 private hands into fixed-size buckets that are useful for dense solving and leaf evaluation.

Checklist:

- [ ] keep the current postflop strength bucketer;
- [ ] add a preflop bucketer with fixed bucket count;
- [ ] define how dead cards are masked in both phases;
- [ ] decide whether preflop buckets are hand-class based, equity based, or hybrid;
- [ ] keep bucket outputs dense and fixed-size.

Key decision:

- use separate bucket logic for preflop and postflop.

Why:

- preflop and postflop answer different questions;
- postflop strength depends on the board and rank class;
- preflop bucketing should be stable before any board is dealt.

Tests to add or update after Phase 3:

- postflop bucket assignment still matches evaluator rank classes;
- preflop bucket assignment is deterministic and covers all valid hands;
- bucket masking marks blocked hands correctly;
- bucketed range sums remain stable and normalized where expected;
- blocked hands never leak into active buckets.

## Phase 4 - Connect Hold'em to depth-limited solving

Goal:

- make the Hold'em tree usable by the existing CFR solver stages.

Checklist:

- [ ] map Hold'em tree nodes to infosets in the same dense layout used elsewhere;
- [ ] verify leaf cutoff feeds into the leaf-evaluation pipeline;
- [ ] keep forward reach, backward CFV, and regret update contracts intact;
- [ ] confirm board-aware leaf features are available at the frontier;
- [ ] make sure the solver still works with toy games after the Hold'em wiring is added.

Key decision:

- depth-limited leaves should be first-class, not a special-case hack.

Why:

- the later GPU value network depends on the same leaf interface;
- special-case leaf handling tends to break when the tree grows.

Tests to add or update after Phase 4:

- depth-limited Hold'em solve reaches the expected frontier;
- leaf evaluator receives the correct input shape and board context;
- stage-by-stage solver tests still pass on Kuhn and Leduc;
- root strategy output remains valid probability mass.

## Phase 5 - Add Hold'em-specific regression tests

Goal:

- lock down behavior before optimizing further.

Checklist:

- [ ] tree construction regression tests;
- [ ] legal-action regression tests;
- [ ] bucket regression tests;
- [ ] depth-limit regression tests;
- [ ] solver smoke test for a minimal HU Hold'em state.

Recommended test focus:

- output shape;
- legality;
- determinism;
- no duplicate actions;
- no invalid card overlaps;
- no regression in Kuhn/Leduc paths.

Tests to check after Phase 5:

- all new Hold'em tests pass;
- the existing solver suite still passes;
- benchmark or smoke scripts still run on toy games;
- no new mypy or lint issues are introduced in hot-path code.

## Phase 6 - Integrate GPU leaf evaluation for Hold'em

Goal:

- make the Hold'em HU depth-limited path benefit from the existing GPU branch.

Checklist:

- [ ] confirm leaf feature packing works for Hold'em board states;
- [ ] validate CPU and GPU leaf backends produce equivalent outputs on shared fixtures;
- [ ] measure batch size behavior for Hold'em leaves;
- [ ] keep CPU-GPU transfer bounded to the leaf frontier;
- [ ] avoid dynamic shapes in hot GPU paths where possible.

Key decision:

- the GPU should only evaluate batched leaves, not drive tree traversal.

Why:

- traversal is branchy and easier to keep on CPU for the first implementation;
- GPU wins are largest when leaf evaluation is batched and stable.

Tests to add or update after Phase 6:

- CPU and GPU leaf parity tests for Hold'em fixtures;
- batch assembly tests for frontier leaves;
- smoke test that the solver still completes with the GPU backend enabled;
- benchmark check that GPU integration does not regress correctness.

## Phase 7 - Optimize and refine abstraction

Goal:

- improve the quality of the Hold'em abstraction after the first working version is stable.

Checklist:

- [ ] review preflop bucket quality;
- [ ] review postflop bucket quality;
- [ ] review bet-size granularity;
- [ ] consider street-specific action refinement;
- [ ] measure whether the chosen depth limit is useful in practice;
- [ ] tune for benchmarked latency and solver quality.

Tests to check after Phase 7:

- benchmark comparisons across abstraction changes;
- solver-quality regression checks;
- no loss of tree determinism;
- no breakage in the existing toy-game baselines.

## Suggested execution order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

## Practical note

If we want fast progress, implement and verify one phase at a time. The most important early proof is:

- Hold'em HU tree builds successfully;
- actions are legal and deterministic;
- buckets are stable;
- depth limits produce the expected leaves;
- Kuhn and Leduc remain green.

