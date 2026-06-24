# Heads-Up Hold'em Full Tree Checklist

Use this as the step-by-step execution checklist.

## Step 0: Baseline snapshot

- [ ] Record current tree diagnostics for the Hold'em HU solver
- [ ] Record current root branching and depth-limit behavior
- [ ] Confirm which CLI path builds the tree used by diagnostics

Files:

- [`src/pokergpu/cfr/solver/service.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/service.py)
- [`src/pokergpu/cfr/solver/tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py)

## Step 1: Tree data model

- [ ] Add support for chance node representation if missing
- [ ] Add chance edge probabilities if missing
- [ ] Keep the tree structure compact and typed

Files:

- [`src/pokergpu/tree/public_tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/public_tree.py)
- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)

## Step 2: Builder behavior

- [ ] Expand the builder beyond player-only traversal
- [ ] Insert chance nodes for card dealing
- [ ] Advance streets correctly after betting rounds end
- [ ] Stop using a single fixed board path as the only future board

Files:

- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)
- [`src/pokergpu/core/transitions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/transitions.py)

## Step 3: Hold'em root

- [ ] Build a proper canonical heads-up root state
- [ ] Separate chance dealing from betting nodes
- [ ] Preserve street progression from preflop to river

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

