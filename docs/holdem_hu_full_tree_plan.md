# Heads-Up Hold'em Full Tree Plan

Goal: replace the current tiny, depth-capped public tree with a realistic heads-up no-limit Texas Hold'em tree that includes chance nodes, street progression, and a richer infoset/action structure.

## Current problem

The current tree is not a real Hold'em tree yet:

- tree depth is capped very aggressively
- board progression is deterministic instead of chance-based
- the tree only expands a small number of player actions
- infosets are assigned as a dense counter, not by poker state identity
- the resulting diagnostics are too small for meaningful CFR training

## Target outcome

We want a tree that:

- starts from a proper preflop chance state
- includes hole-card dealing and community-card chance nodes
- expands through preflop, flop, turn, and river betting rounds
- supports legal no-limit actions, including fold, check, call, bet, raise, and all-in behavior
- creates infosets from player position, private hand, and public state
- produces nontrivial tree diagnostics:
  - chance nodes > 0
  - infosets >> 4
  - internal nodes >> 4
  - branching varies by street and state

## Recommended implementation path

1. Fix tree construction so it can represent chance nodes.
2. Build a real heads-up Hold'em public tree from a canonical root state.
3. Expand infoset identity to reflect poker state, not just traversal order.
4. Update diagnostics and tests.
5. Only after the tree is correct, tune abstraction sizes and depth limits.

## Files to change

- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)
- [`src/pokergpu/tree/public_tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/public_tree.py)
- [`src/pokergpu/cfr/solver/tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py)
- [`src/pokergpu/cfr/solver/spec.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/spec.py)
- [`src/pokergpu/abstraction/actions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/actions.py)
- [`src/pokergpu/cfr/solver/service.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/service.py)
- [`tests/`](/C:/Users/xeuse/RiderProjects/PokerGPU/tests)

## Phase breakdown

### Phase 1: Make chance explicit

What to do:

- add a way for the tree builder to emit chance nodes
- represent chance edges with probabilities
- model hole-card dealing and street deal cards

Why:

- without explicit chance, the game tree is structurally incorrect
- CFR reach propagation and EV calculations depend on chance branching

### Phase 2: Fix hold'em root construction

What to do:

- start from a canonical preflop public state
- create a chance root that deals private cards before player decisions
- advance streets using real board transitions, not one fixed sample board

Why:

- a full Hold'em tree must include actual community-card uncertainty

### Phase 3: Expand legal action modeling

What to do:

- verify betting abstraction returns all intended legal actions
- ensure raise-to sizing works across streets and stack sizes
- handle check/call/fold/all-in edge cases consistently

Why:

- the tree should reflect the actual no-limit game rules, even if abstracted

### Phase 4: Make infosets poker-state aware

What to do:

- derive infoset identity from:
  - acting player
  - street
  - public board texture
  - private hand class or exact hand, depending on abstraction level

Why:

- a dense traversal counter collapses distinct poker situations into one bucket

### Phase 5: Add diagnostics and tests

What to do:

- assert chance node counts are nonzero
- assert node count and infoset count grow beyond trivial values
- assert street progression reaches flop, turn, and river in the full tree path
- assert labels and branching differ by street

Why:

- this prevents regressions back to the tiny diagnostic tree

## Acceptance criteria

- `tree_chance_nodes` is greater than `0`
- `tree_infosets` is meaningfully larger than `4`
- `tree_internal_nodes` is meaningfully larger than `4`
- the root is not a single fixed board path
- tests cover the canonical HU Hold'em tree structure

