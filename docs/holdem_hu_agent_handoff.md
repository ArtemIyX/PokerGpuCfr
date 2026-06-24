# Heads-Up Hold'em Agent Handoff

This document is for LLM agents working the tree fix in small steps.

## Mission

Turn the current Hold'em HU solver tree into a realistic public game tree suitable for CFR experiments.

## Important constraints

- Do not mask the problem by lowering depth again.
- Do not remove chance nodes if they are added.
- Do not change the solver to fit the tiny tree.
- Fix the tree model instead of weakening diagnostics.
- Keep changes small and test-driven.

## Suggested work order

### Agent 1: Tree structure

Primary goal:

- make the tree builder capable of chance nodes and street progression

Files to inspect first:

- [`src/pokergpu/tree/builder.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/builder.py)
- [`src/pokergpu/tree/public_tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/tree/public_tree.py)

Deliverables:

- tree can represent chance nodes explicitly
- no tree invariants are broken
- tests cover the new representation

### Agent 2: Hold'em root and dealing

Primary goal:

- build a canonical HU Hold'em root with real dealing branches

Files to inspect first:

- [`src/pokergpu/cfr/solver/tree.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/tree.py)
- [`src/pokergpu/core/transitions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/core/transitions.py)

Deliverables:

- preflop dealing is not collapsed into one path
- flop/turn/river are reachable through chance logic

### Agent 3: Action abstraction

Primary goal:

- ensure the action set is realistic enough for heads-up no-limit play

Files to inspect first:

- [`src/pokergpu/abstraction/actions.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/abstraction/actions.py)

Deliverables:

- legal actions are street-aware
- action labels and action counts match the tree

### Agent 4: Infosets and diagnostics

Primary goal:

- make infosets reflect poker state and expose better diagnostics

Files to inspect first:

- [`src/pokergpu/cfr/solver/infosets.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/infosets.py)
- [`src/pokergpu/cfr/solver/service.py`](/C:/Users/xeuse/RiderProjects/PokerGPU/src/pokergpu/cfr/solver/service.py)

Deliverables:

- infosets are no longer trivial
- diagnostics show nonzero chance nodes and higher counts

## What success looks like

- root diagnostics show a genuine poker tree, not a toy tree
- chance nodes exist
- infosets grow with board and action complexity
- tests fail if the tree regresses to the current tiny structure

