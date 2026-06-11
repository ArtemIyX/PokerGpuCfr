# PLAN.md

Purpose:
- Track the remaining work for the GPU poker solver.
- Keep the plan aligned with what already exists in the repo.
- Focus on MVP: abstracted 6-max NLHE, offline blueprint, runtime re-solving, CUDA leaf eval.

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

## 2. Lock game representation

- [x] Verify every action updates state correctly
- [x] Verify legal action generation for each street
- [x] Verify terminal detection for folds, calls, and showdowns
- [x] Verify payout calculation for all side-pot cases
- [x] Verify board progression and street transitions
- [x] Add regression tests for edge-case hands

## 3. Finish action abstraction

- [x] Define fixed bet sizes by street
- [x] Split abstraction by position group
- [x] Handle min-raise and all-in rules
- [x] Keep action sets consistent across tree builds

## 4. Add private-hand indexing and range vectors

- [x] Build canonical private-hand IDs
- [x] Add range vector storage per player
- [x] Add dead-card masking and normalization
- [x] Verify range sums stay valid after board cards

## 5. Add postflop bucketing and suit-isomorphic canonicalization

- [x] Define bucket features for hand strength and draw value
- [x] Add canonical board normalization
- [x] Share canonicalization across tree, ranges, and runtime
- [x] Add tests for isomorphic board equivalence

## 6. Build dense infoset/regret/strategy storage

- [x] Replace dynamic lookup paths in hot code
- [x] Use flat offsets for regrets and strategy sums
- [x] Precompute infoset to action-count mapping
- [x] Keep all solver tables contiguous

## 7. Implement full CFR traversal

- [ ] Traverse the public tree forward and backward
- [ ] Compute counterfactual values for each infoset
- [ ] Update regrets and average strategy
- [ ] Add CFR+ and DCFR support

## 8. Validate on toy games

- [ ] Run Kuhn end to end
- [ ] Run Leduc end to end
- [ ] Compare strategies and values against expected baselines
- [ ] Fix convergence and stability issues

## 9. Add heads-up postflop re-solving

- [ ] Build depth-limited subtree solving
- [ ] Add leaf evaluation at frontier nodes
- [ ] Warm start from cached blueprint state
- [ ] Return root strategy for real spots

## 10. Add leaf batching and a trained value model

- [ ] Define leaf feature tensors
- [ ] Build training data from solved subgames
- [ ] Train a small value model first
- [ ] Add GPU inference for batch leaf eval

## 11. Extend to cached blueprint solving and multiway 6-max approximation

- [ ] Add blueprint export and reload
- [ ] Add subtree caching and warm starts
- [ ] Approximate multiway as restricted subgames
- [ ] Measure solver quality and speed
