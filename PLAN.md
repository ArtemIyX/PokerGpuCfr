# PLAN.md

Purpose:
- Define the full build plan for this Python GPU-CFR poker project.
- Keep progress visible with checkboxes.
- Update this file as tasks are completed or split further.

How to use:
- `[ ]` not started
- `[-]` in progress
- `[x]` done
- Add short notes under a task when needed.

---

## 1. Project foundation

- [x] Define package layout under `src/`
- [x] Add runnable entrypoint for local development
- [x] Add config system for paths, device, and solver settings
- [x] Add logging setup
- [x] Add test setup
- [x] Add formatting, lint, and type-check tooling
- [x] Add benchmark harness scaffold

## 2. Core game model

- [x] Implement core poker types: cards, suits, ranks
- [x] Implement deck generation and shuffle utilities
- [x] Implement hand parsing and formatting
- [x] Implement board representation
- [x] Implement stack, pot, blind, and bet state types
- [x] Implement NLHE betting rules
- [x] Implement action legality checks
- [x] Add unit tests for rules and edge cases

## 3. Hand evaluation layer

- [x] Choose evaluator approach for Python project
- [x] Implement 5-card hand evaluation
- [x] Implement 7-card hand evaluation
- [x] Add batch evaluation API
- [x] Add correctness tests against known hand rankings
- [x] Add performance benchmark for evaluator

## 4. Game state and transitions

- [x] Implement immutable or controlled-mutation game state model
- [x] Implement action apply/undo or state transition engine
- [x] Implement terminal detection
- [x] Implement payout computation
- [x] Implement public-state signature
- [x] Add tests for state transitions and payouts

## 5. Public tree representation

- [x] Define flat-array public tree data model
- [x] Implement node indexing and child ranges
- [x] Implement terminal and chance node encoding
- [x] Implement infoset references from public nodes
- [x] Add tree validation checks
- [x] Add tests for tree structure integrity

## 6. Action abstraction

- [x] Define abstraction interface
- [x] Implement baseline bet sizing set
- [x] Implement min-raise and stack-aware sizing filters
- [x] Implement street-specific action templates
- [x] Implement configurable abstraction profiles
- [x] Add tests for legal generated actions

## 7. Card abstraction and ranges

- [x] Define private-hand indexing scheme
- [x] Implement range representation API
- [x] Implement range normalization and masking
- [x] Implement postflop bucket interface
- [x] Start with simple bucket baseline
- [x] Add tests for range math and masking

## 8. Tiny CFR baseline

- [x] Implement dense-array infoset storage
- [x] Implement regret matching
- [x] Implement CFR iteration loop
- [x] Implement average strategy accumulation
- [x] Implement exploitability or sanity metrics for toy games
- [x] Add Kuhn Poker implementation
- [x] Validate convergence on Kuhn Poker
- [x] Add Leduc Poker implementation
- [x] Validate convergence on Leduc

## 9. CFR variants

- [x] Implement CFR+
- [x] Implement DCFR
- [x] Make solver variant configurable
- [x] Compare convergence on toy games
- [x] Record benchmark results


## 10. Parallel CPU solver core

- [x] Refactor traversal around flat arrays
- [x] Separate forward pass, backward pass, and regret update
- [x] Add parallel infoset updates
- [x] Add parallel node traversal where useful
- [x] Add deterministic reduction strategy
- [x] Benchmark single-thread vs multi-thread

## 11. Depth-limited solving

- [x] Define depth limit policy
- [x] Mark frontier nodes for leaf evaluation
- [x] Add configurable node-count cap
- [x] Add configurable reach-probability pruning
- [x] Add tests for frontier correctness

## 12. Leaf evaluation interface

- [x] Define leaf feature schema
- [x] Define evaluator interface returning EVs
- [x] Implement CPU stub evaluator
- [x] Integrate evaluator into CFR loop
- [x] Add tests for batch build and scatter logic

## 13. GPU path for batched evaluation

- [x] Choose first GPU backend for Python
- [x] Implement device selection and fallback
- [x] Implement batch tensor builder
- [x] Implement GPU inference stub
- [x] Add async or overlapped batch execution design
- [x] Measure batch throughput
- [x] Verify numeric parity with CPU stub where applicable

## 14. Postflop re-solving prototype

- [x] Limit scope to heads-up postflop first
- [x] Build public subtree from current state
- [x] Support player ranges at root
- [x] Run time-budgeted solving loop
- [x] Return root mixed strategy
- [x] Add CLI or script demo for one postflop spot
  - Added `postflop-resolve` CLI path and runtime resolver module.

## 15. Caching and warm start

- [x] Define public-state cache key
- [x] Cache subtree structures
- [x] Cache leaf evaluation results when reusable
- [x] Add regret warm-start support
- [x] Benchmark cold vs warm solve

