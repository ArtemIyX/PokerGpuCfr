# Hold'em HU Implementation Plan

This document is the canonical implementation roadmap for heads-up no-limit Texas Hold'em in PokerGPU.

The goal is not an exact full-game solver in phase 1. The goal is a practical, depth-limited, GPU-friendly heads-up path that plugs into the existing CFR pipeline without destabilizing Kuhn or Leduc.

We want the first implementation to be:

- deterministic;
- shallow enough to validate;
- easy to inspect with tests;
- compatible with the current flat public-tree and batched leaf-eval design;
- expandable toward a fuller Hold'em abstraction later.

## Scope

### In scope for this plan

- heads-up only, 1 vs 1;
- no-limit betting abstraction with a fixed global action layout;
- street-aware betting templates;
- depth-limited solving;
- preflop and postflop card abstraction;
- board-aware leaf features;
- CPU correctness first, GPU integration second;
- regression tests for each phase.

### Out of scope for the first pass

- full exact NLHE tree enumeration;
- multiway games;
- 6-max support;
- advanced pruning;
- learned abstraction;
- exact combinatorial board dealing at every public node;
- perfect-information full-depth solve of the entire game.

## Existing Building Blocks

The repository already contains most of the infrastructure needed for Hold'em:

- `Board` already defines street boundaries by board length in [src/pokergpu/core/board.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/board.py).
- `BettingRoundState` already stores pot, stacks, bets, blinds, and the acting player in [src/pokergpu/core/betting.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/betting.py).
- legality helpers already exist in [src/pokergpu/core/legality.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/legality.py).
- min/max raise sizing already exists in [src/pokergpu/core/rules.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/rules.py).
- the action abstraction layer is already street-aware in [src/pokergpu/abstraction/actions.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/actions.py).
- the 1326-combo private-hand abstraction exists in [src/pokergpu/abstraction/hands.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/hands.py).
- postflop strength bucketing already exists in [src/pokergpu/abstraction/buckets.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/buckets.py).
- the CFR pipeline already supports `LEAF` frontier nodes and board-aware leaf evaluation in [src/pokergpu/cfr/stage2.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/stage2.py) and [src/pokergpu/cfr/solver/evaluation.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/evaluation.py).

## Concrete Minimal Tree Shape

The first Hold'em HU scaffold should be shallow, deterministic, and use `LEAF` for depth truncation.

### Core design

- root is preflop after blinds are posted;
- internal nodes alternate between `PLAYER0` and `PLAYER1`;
- fold branches are `TERMINAL`;
- depth cutoff branches are `LEAF`;
- exact showdown or all-in resolution can remain out of the first scaffold if needed;
- street transition can be represented explicitly with future chance nodes, but the first tree does not need full board enumeration.

### Recommended minimal scaffold

Use this as the first stable tree shape:

- `Node 0`: `PLAYER0` preflop decision node
- `Node 1`: `PLAYER1` response node after player 0 action
- `Node 2`: `TERMINAL` fold payoff for player 0 winning the pot
- `Node 3`: `LEAF` depth-limited frontier for continuation after an action branch
- `Node 4`: `TERMINAL` fold payoff for player 1 winning the pot
- `Node 5`: `LEAF` depth-limited frontier for continuation after a bet/raise branch

This is intentionally small. The exact node count can grow later, but the first version should satisfy:

- non-empty tree;
- at least one player node;
- at least one terminal node;
- at least one leaf node;
- deterministic child ordering;
- explicit infoset assignment for player nodes.

### Recommended first cut location

The first `LEAF` cut should occur at the first unresolved public continuation after the initial preflop betting exchange.

Concretely:

- root preflop action node is expanded;
- opponent response node is expanded;
- any branch that would continue into a richer postflop or deeper betting sequence becomes `LEAF`.

That gives us:

- enough structure to validate action legality;
- a stable hook for leaf evaluation;
- a simple root/frontier shape for debugging.

### Preferred future full-street shape

When the scaffold is expanded, the tree should evolve into this logical sequence:

1. preflop betting;
2. flop deal or flop frontier;
3. flop betting;
4. turn deal or turn frontier;
5. turn betting;
6. river deal or river frontier;
7. river betting;
8. showdown or terminal fold resolution.

For phase 1 and phase 2, we do not need the full sequence. The key is that the implementation should be able to grow toward this shape without changing the solver contract.

## Data Model Decisions

### Public state shape

The public Hold'em state should be representable by the existing core model:

- `Board` for street and community cards;
- `GameState` for players, betting round, phase, and dealer;
- `BettingRoundState` for chips, stacks, bets, blinds, and `to_act`.

### Recommended Hold'em public-state invariants

- exactly two players;
- dealer is one of the two players;
- stacks and bets must align with player ids;
- board length must match street;
- player hole cards must not overlap with board cards;
- preflop root should encode blinds already posted;
- the tree builder should reject invalid states rather than guessing.

### Tree cutoff policy

Use both concepts, but with different roles:

- `LEAF` means depth-limited frontier and must go through the leaf backend;
- `TERMINAL` means exact resolved payoff;
- only `LEAF` should represent the first Hold'em truncation frontier.

## Phase 0 - Lock the Hold'em data model

Goal:

- define the exact minimal public-state shape and tree root assumptions before implementing the solver path.

### Checklist

- [x] confirm heads-up only, exactly two players;
- [x] confirm the root starts from preflop after blinds are posted;
- [ ] confirm dealer/button encoding and who acts first preflop;
- [x] confirm whether the first implementation uses a fixed cutoff depth, a street cutoff, or both;
- [x] confirm whether exact showdowns are handled immediately or deferred to leaf evaluation;
- [x] confirm whether the builder accepts a full `GameState` or a reduced Hold'em-specific public state object;
- [x] confirm which states are valid at the solver entry point and which must be rejected.

### Implementation notes

- use `GameState` and `BettingRoundState` if possible;
- avoid creating a parallel Hold'em state class unless the existing core model cannot express the requirements cleanly;
- treat invalid states as hard errors;
- keep the invariants explicit in tests.

### Tests to write after Phase 0

- minimal HU Hold'em public-state construction test;
- invalid public-state rejection test;
- board/street alignment test;
- dealer validity test;
- player stack / bet alignment test;
- regression test that Kuhn and Leduc state handling still passes.

## Phase 1 - Add Hold'em HU tree scaffolding

Goal:

- make `GameVariant.HOLDEM_HU` build a non-empty public tree.

### Checklist

- [x] add `make_holdem_hu_public_tree`;
- [x] wire `GameVariant.HOLDEM_HU` into the tree factory in [src/pokergpu/cfr/solver/tree.py](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py);
- [x] preserve the flat `PublicTree` representation;
- [x] keep node ordering deterministic;
- [x] include both `TERMINAL` and `LEAF` nodes in the first scaffold;
- [x] keep child counts and `first_child` offsets consistent with the array representation;
- [ ] keep `HOLDEM_6MAX` unsupported for now;
- [ ] make unsupported variants fail with a clear error message.

### Minimal target behavior

- `HOLDEM_HU` returns a valid tree;
- root is a player node;
- root has legal action children;
- at least one branch ends in `TERMINAL`;
- at least one branch ends in `LEAF`;
- the tree remains small enough to inspect in tests.

### Tests to write after Phase 1

- tree factory returns a non-empty Hold'em HU tree;
- root node type and child count are correct;
- `LEAF` nodes exist in the tree;
- `TERMINAL` nodes exist in the tree;
- tree construction is deterministic across repeated calls;
- Kuhn and Leduc trees remain unchanged;
- `HOLDEM_6MAX` still fails explicitly if it remains unsupported.

## Phase 2 - Implement street-aware Hold'em action abstraction

Goal:

- make the global action profile explicitly different by street.

### Checklist

- [ ] define a Hold'em-specific `AbstractionProfile`;
- [ ] keep the action vocabulary fixed across the tree;
- [ ] split preflop and postflop sizing rules;
- [ ] make river sizing different from flop and turn if needed;
- [ ] clamp bet sizes to stack size and minimum legal bet;
- [ ] clamp raise sizes to min/max legal raise-to bounds;
- [ ] eliminate duplicate actions after rounding/clamping;
- [ ] keep the ordering of emitted actions deterministic;
- [ ] keep `check/call/fold` handling legality-driven;
- [ ] preserve support for the current toy-game tests.

### Recommended street layout

Use one global vocabulary, but different street templates:

- preflop:
  - smaller template
  - fewer bet sizes
  - possibly fewer raise-to sizes
  - more conservative sizing because pot-relative sizing is weak preflop
- flop:
  - wider pot-relative template
  - at least 3 bet sizes
- turn:
  - similar to flop, possibly slightly reduced
- river:
  - pot-relative template with emphasis on value-heavy sizes

### Tests to write after Phase 2

- preflop action enumeration includes the intended legal subset;
- flop action enumeration includes the intended percentage sizes;
- turn action enumeration differs from preflop if the profile says it should;
- river action enumeration differs from flop if the profile says it should;
- facing-bet states return fold/call/raise only when legal;
- unopened-pot states return check/bet only when legal;
- stack clamping works for small stacks;
- minimum sizing works when pot-relative rounding underflows;
- duplicate actions are not emitted.

## Phase 3 - Add preflop and postflop card abstraction

Goal:

- reduce 1326 private hands into stable fixed-size buckets suitable for dense CFR and leaf features.

### Checklist

- [ ] keep the current postflop strength bucketer;
- [ ] add a preflop bucketer;
- [ ] define a fixed bucket count for preflop;
- [ ] decide whether preflop buckets are class-based, equity-based, or hybrid;
- [ ] define how dead cards are masked preflop and postflop;
- [ ] ensure bucket outputs are dense and fixed-width;
- [ ] keep bucket assignment deterministic;
- [ ] keep bucketed range outputs normalized where expected;
- [ ] make blocked hands map to a sentinel or zeroed bucket contribution.

### Recommended abstraction split

- preflop:
  - hand-class or equity-based buckets;
  - fixed width;
  - no board dependency;
- postflop:
  - keep the existing `StrengthTierBucketer`;
  - board-dependent;
  - fixed width `9`.

### Tests to write after Phase 3

- preflop bucketer is deterministic;
- preflop bucketer covers all valid hands;
- preflop bucketer rejects invalid or blocked combos as designed;
- postflop bucketer still rejects preflop boards;
- postflop bucket assignment still matches evaluator rank classes;
- bucket masking marks blocked hands correctly;
- bucketed ranges sum to the expected total weight;
- blocked hands do not leak into active buckets.

## Phase 4 - Connect Hold'em to the CFR pipeline

Goal:

- make the Hold'em tree usable by the existing stage-based solver.

### Checklist

- [ ] map Hold'em tree nodes to infosets using the same dense layout as toy games;
- [ ] verify depth cutoff nodes are treated as `LEAF`;
- [ ] ensure leaf features include board context;
- [ ] keep forward reach propagation compatible with the Hold'em tree shape;
- [ ] keep backward CFV and regret update contracts intact;
- [ ] confirm the solver still works with Kuhn and Leduc after Hold'em wiring;
- [ ] make sure root strategy output remains a valid probability distribution.

### Important implementation rule

- do not special-case Hold'em deep inside stage 1 to stage 7;
- instead, make the tree and abstractions conform to the existing pipeline contracts.

### Tests to write after Phase 4

- depth-limited Hold'em solve reaches the expected frontier;
- leaf evaluator receives the expected input width;
- leaf features include board/street metadata;
- stage-by-stage solver smoke tests still pass on Kuhn and Leduc;
- root strategy output remains normalized;
- stage 2 finds Hold'em leaf nodes correctly;
- stage 6 can backpropagate through Hold'em leaves.

## Phase 5 - Add Hold'em regression tests

Goal:

- lock down behavior before optimization and GPU tuning.

### Checklist

- [ ] tree construction regression tests;
- [ ] legal-action regression tests;
- [ ] bucket regression tests;
- [ ] depth-limit regression tests;
- [ ] solver smoke test for a minimal HU Hold'em state;
- [ ] invalid-state regression tests;
- [ ] determinism regression tests;
- [ ] no-duplicate-action regression tests;
- [ ] no invalid card overlap regression tests.

### Tests to write after Phase 5

- all new Hold'em tests pass;
- existing toy-game tests remain green;
- the Hold'em tree is deterministic under repeated construction;
- the Hold'em solver smoke test completes successfully;
- invalid states fail fast and clearly.

## Phase 6 - Integrate GPU leaf evaluation for Hold'em

Goal:

- make the Hold'em depth-limited path benefit from the current GPU leaf backend.

### Checklist

- [ ] confirm leaf feature packing works for Hold'em boards;
- [ ] validate CPU and GPU leaf backends on shared fixtures;
- [ ] measure batch size behavior for frontier leaves;
- [ ] keep CPU-GPU transfer limited to the leaf frontier;
- [ ] avoid dynamic shapes in hot GPU paths where possible;
- [ ] confirm fallback CPU backend still works when GPU is unavailable;
- [ ] preserve deterministic node ordering in batched leaf outputs.

### Tests to write after Phase 6

- CPU and GPU leaf parity tests for Hold'em fixtures;
- batched frontier assembly tests;
- solver smoke test with GPU backend enabled;
- solver smoke test with heuristic backend enabled;
- leaf ordering preservation test;
- correctness regression test comparing CPU and GPU output shapes.

## Phase 7 - Optimize and refine the abstraction

Goal:

- improve abstraction quality after the first working version is stable.

### Checklist

- [ ] review preflop bucket quality;
- [ ] review flop, turn, and river bucket quality;
- [ ] review bet-size granularity;
- [ ] consider street-specific action refinement;
- [ ] measure whether the chosen depth limit is useful in practice;
- [ ] tune for latency and solver quality;
- [ ] revisit leaf feature design if the value backend underfits;
- [ ] reconsider tree breadth if the action profile is too coarse.

### Tests to write after Phase 7

- benchmark comparisons across abstraction changes;
- solver quality regression checks;
- tree determinism checks;
- no breakage in toy-game baselines;
- no regression in GPU leaf parity.

## Suggested Execution Order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

## Practical Implementation Notes

- implement one phase at a time;
- keep the first Hold'em tree small enough to inspect manually;
- prefer hard validation over silent fallback;
- keep Kuhn and Leduc passing while adding Hold'em;
- do not widen the tree or abstraction until the current shape is covered by tests;
- use `LEAF` for the depth frontier and `TERMINAL` for exact payoffs;
- preserve the flat array tree layout so the solver pipeline remains unchanged;
- if a design choice is unclear, encode it explicitly in tests before expanding implementation.

## Minimal First Deliverable

The first acceptable Hold'em HU milestone is:

- `GameVariant.HOLDEM_HU` returns a valid public tree;
- the tree is shallow, deterministic, and includes `LEAF` frontier nodes;
- the action abstraction is street-aware;
- the solver can run a smoke test on the Hold'em tree;
- Kuhn and Leduc still pass unchanged.
