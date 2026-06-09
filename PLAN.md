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

- [ ] Implement immutable or controlled-mutation game state model
- [ ] Implement action apply/undo or state transition engine
- [ ] Implement terminal detection
- [ ] Implement payout computation
- [ ] Implement public-state signature
- [ ] Add tests for state transitions and payouts

## 5. Public tree representation

- [ ] Define flat-array public tree data model
- [ ] Implement node indexing and child ranges
- [ ] Implement terminal and chance node encoding
- [ ] Implement infoset references from public nodes
- [ ] Add tree validation checks
- [ ] Add tests for tree structure integrity

## 6. Action abstraction

- [ ] Define abstraction interface
- [ ] Implement baseline bet sizing set
- [ ] Implement min-raise and stack-aware sizing filters
- [ ] Implement street-specific action templates
- [ ] Implement configurable abstraction profiles
- [ ] Add tests for legal generated actions

## 7. Card abstraction and ranges

- [ ] Define private-hand indexing scheme
- [ ] Implement range representation API
- [ ] Implement range normalization and masking
- [ ] Implement postflop bucket interface
- [ ] Start with simple bucket baseline
- [ ] Add tests for range math and masking

## 8. Tiny CFR baseline

- [ ] Implement dense-array infoset storage
- [ ] Implement regret matching
- [ ] Implement CFR iteration loop
- [ ] Implement average strategy accumulation
- [ ] Implement exploitability or sanity metrics for toy games
- [ ] Add Kuhn Poker implementation
- [ ] Validate convergence on Kuhn Poker
- [ ] Add Leduc Poker implementation
- [ ] Validate convergence on Leduc

## 9. CFR variants

- [ ] Implement CFR+
- [ ] Implement DCFR
- [ ] Make solver variant configurable
- [ ] Compare convergence on toy games
- [ ] Record benchmark results

## 10. Parallel CPU solver core

- [ ] Refactor traversal around flat arrays
- [ ] Separate forward pass, backward pass, and regret update
- [ ] Add parallel infoset updates
- [ ] Add parallel node traversal where useful
- [ ] Add deterministic reduction strategy
- [ ] Benchmark single-thread vs multi-thread

## 11. Depth-limited solving

- [ ] Define depth limit policy
- [ ] Mark frontier nodes for leaf evaluation
- [ ] Add configurable node-count cap
- [ ] Add configurable reach-probability pruning
- [ ] Add tests for frontier correctness

## 12. Leaf evaluation interface

- [ ] Define leaf feature schema
- [ ] Define evaluator interface returning EVs
- [ ] Implement CPU stub evaluator
- [ ] Integrate evaluator into CFR loop
- [ ] Add tests for batch build and scatter logic

## 13. GPU path for batched evaluation

- [ ] Choose first GPU backend for Python
- [ ] Implement device selection and fallback
- [ ] Implement batch tensor builder
- [ ] Implement GPU inference stub
- [ ] Add async or overlapped batch execution design
- [ ] Measure batch throughput
- [ ] Verify numeric parity with CPU stub where applicable

## 14. Postflop re-solving prototype

- [ ] Limit scope to heads-up postflop first
- [ ] Build public subtree from current state
- [ ] Support player ranges at root
- [ ] Run time-budgeted solving loop
- [ ] Return root mixed strategy
- [ ] Add CLI or script demo for one postflop spot

## 15. Caching and warm start

- [ ] Define public-state cache key
- [ ] Cache subtree structures
- [ ] Cache leaf evaluation results when reusable
- [ ] Add regret warm-start support
- [ ] Benchmark cold vs warm solve

## 16. Data generation for learned value model

- [ ] Define training sample schema
- [ ] Generate solved or partially solved targets
- [ ] Store datasets on disk in efficient format
- [ ] Add dataset inspection script
- [ ] Add reproducible data config

## 17. Neural value model

- [ ] Choose framework for first model
- [ ] Implement model input pipeline
- [ ] Implement baseline network
- [ ] Train on generated targets
- [ ] Evaluate calibration and EV error
- [ ] Export inference artifact

## 18. GPU leaf evaluation with trained model

- [ ] Replace stub with trained model inference
- [ ] Optimize batch size and precision
- [ ] Add mixed-precision inference if stable
- [ ] Measure end-to-end solve improvement
- [ ] Track GPU utilization during solve

## 19. Scaling toward larger games

- [ ] Expand abstraction quality for HUNL postflop
- [ ] Add turn and river support details
- [ ] Improve bucket quality
- [ ] Add pruning heuristics
- [ ] Add solver quality regression suite

## 20. Multiway path

- [ ] Define approximate multiway design
- [ ] Decide coalition/nested-subgame approach
- [ ] Prototype 3-player restricted solve
- [ ] Evaluate complexity and memory impact
- [ ] Decide whether full 6-max path remains practical

## 21. Tooling and interfaces

- [ ] Add CLI commands for build, solve, benchmark, and train
- [ ] Add config files for common scenarios
- [ ] Add artifact directories and naming conventions
- [ ] Add result serialization
- [ ] Add plotting/report scripts

## 22. Validation and benchmarks

- [ ] Define benchmark scenarios
- [ ] Measure toy-game convergence
- [ ] Measure CPU scaling
- [ ] Measure GPU batch speedup
- [ ] Measure end-to-end re-solve latency
- [ ] Document quality vs speed tradeoffs

## 23. Documentation

- [ ] Keep `PROJECT_LOG.md` updated after every task
- [ ] Keep this `PLAN.md` updated as work progresses
- [ ] Add architecture doc for package layout
- [ ] Add developer setup doc
- [ ] Add solver workflow doc
- [ ] Add benchmark report doc

## 24. Definition of done

- [ ] Toy CFR baselines are correct and tested
- [ ] Flat-array CPU solver is stable
- [ ] GPU leaf batching is integrated
- [ ] Heads-up postflop re-solver works within target time budget
- [ ] Benchmarks prove meaningful GPU advantage
- [ ] Docs are enough for another engineer or LLM to continue
