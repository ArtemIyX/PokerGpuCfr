# Heads-Up Hold'em Full Tree Checklist

Use this as the step-by-step execution checklist.

## Current status

- [x] Replace placeholder root-only chance outcomes with a street-aware chance expander
- [x] Expand preflop to sampled flop boards
- [x] Expand flop to sampled turn boards
- [x] Expand turn to sampled river boards
- [x] Remove the deterministic `_next_holdem_board` path
- [x] Increase the Hold'em non-compact depth budget so later street chance layers can appear
- [x] Confirm the test suite still passes after the tree expansion change

Next step:

- add diagnostics and regression tests that verify the new chance layers, street progression, and node counts do not collapse back to the tiny tree

## Step 0: Baseline snapshot

- [x] Record current tree diagnostics for the Hold'em HU solver
- [x] Record current root branching and depth-limit behavior
- [x] Confirm which CLI path builds the tree used by diagnostics

Files:

- [`src/pokergpu/cfr/solver/service.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/service.py)
- [`src/pokergpu/cfr/solver/tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py)

## Step 1: Tree data model

- [x] Add support for chance node representation if missing
- [x] Add chance edge probabilities if missing
- [x] Keep the tree structure compact and typed

Files:

- [`src/pokergpu/tree/public_tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/public_tree.py)
- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)

## Step 2: Builder behavior

- [x] Expand the builder beyond player-only traversal
- [x] Insert chance nodes for card dealing
- [x] Advance streets correctly after betting rounds end
- [x] Stop using a single fixed board path as the only future board

Files:

- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)
- [`src/pokergpu/core/transitions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/transitions.py)

## Step 3: Hold'em root

- [x] Build a proper canonical heads-up root state
- [x] Separate chance dealing from betting nodes
- [x] Preserve street progression from preflop to river

Files:

- [`src/pokergpu/cfr/solver/tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py)

## Step 4: Action abstraction

- [ ] Verify action lists on each street
- [ ] Ensure no-limit sizing is legal under stack and pot constraints
- [ ] Make sure labels reflect the action set used in the tree

Files:

- [`src/pokergpu/abstraction/actions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/actions.py)

## Step 5: Infosets

- [ ] Replace traversal-only infoset assignment with poker-state identity
- [ ] Keep infosets dense and stable
- [ ] Ensure different public states do not collide

Files:

- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)
- [`src/pokergpu/cfr/solver/infosets.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/infosets.py)

## Step 6: Diagnostics

- [ ] Report chance-node counts
- [ ] Report street distribution
- [ ] Report infoset count and branching by street
- [ ] Validate the new tree is no longer trivial

Files:

- [`src/pokergpu/cfr/solver/service.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/service.py)

## Step 7: Tests

- [ ] Add a test that fails on the current tiny tree
- [ ] Add tests for street progression and chance nodes
- [ ] Add tests for action-label stability
- [ ] Add tests for infoset growth

Files:

- [`tests/`](/C:/Users/xeuse/RiderProjects/PokerGPU/tests)
