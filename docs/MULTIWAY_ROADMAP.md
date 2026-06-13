# Multiway Roadmap

Goal: move the current heads-up postflop solver toward practical 3-player and then 4-6 player support.

This is not a full exact 6-max solver plan.
It is a staged roadmap for usable approximation.

## Scope

- Preflop is out of scope for now.
- Heads-up postflop stays as the baseline.
- 3-player postflop is the first multiway target.
- 4-6 player spots use approximation and pruning.
- Value network training uses approximate labels when exact labels do not exist.

## Global Strategy

1. Generalize the solver core from 2 players to `N` active players.
2. Keep the public tree flat and batch-friendly.
3. Use aggressive abstraction.
4. Solve 3-way as the main exact-ish multiway case.
5. Use coalition approximation and a value network for 4-6 way.
6. Validate with sampled best response and monotonicity checks.

## Phase 0: Baseline Audit

Goal: make the current HU postflop code ready for extension.

- [x] Identify all hardcoded 2-player assumptions.
- [x] List all data structures that assume one opponent.
- [x] Confirm tree builder supports variable player count.
- [x] Confirm evaluator API can return values for multiple active players.
- [x] Confirm runtime resolver can carry multiple range vectors.
- [x] Confirm tests cover HU postflop end to end.

Done when:

- [ ] all 2-player assumptions are isolated
- [x] the extension points are explicit
- [ ] HU behavior still matches current output

## Phase 1: Core Multiway Data Model

Goal: make the solver data structures player-count aware.

- [x] Replace single-opponent reach/value logic with per-player arrays.
- [x] Represent active players as a compact list in each node.
- [x] Store one range vector per active player.
- [x] Store one counterfactual value vector per active player.
- [x] Keep flat arrays, not pointer graphs.
- [x] Update terminal payoff handling for `N` players.
- [ ] Update card removal for every active player.

Done when:

- [x] a public tree can be built for 2, 3, 4, 5, and 6 active players
- [x] a leaf evaluation call can return `N` values
- [x] ranges stay normalized after board changes

## Phase 2: 3-Player Postflop

Goal: support the easiest useful multiway case.

Approach:

- exact or near-exact on a heavily abstracted tree
- limited bet sizes
- bucketed hands
- depth-limited re-solving

Checklist:

- [ ] Generalize CFR traversal to 3 active players.
- [ ] Generalize regret updates to `N` players.
- [ ] Generalize average strategy storage.
- [ ] Add 3-way action abstraction.
- [ ] Add 3-way range propagation.
- [ ] Add 3-way leaf evaluation format.
- [ ] Add 3-way runtime resolve path.
- [ ] Add 3-way benchmark spots.

Done when:

- [ ] 3-way postflop can be solved end to end
- [ ] runtime returns an action distribution
- [ ] the solve finishes inside a bounded time budget

## Phase 3: 3-Player Quality Controls

Goal: make 3-way output stable enough to trust.

- [ ] Compare 3-way results against simplified baselines.
- [ ] Run sampled best-response checks.
- [ ] Track exploitability trend over iterations.
- [ ] Verify monotonicity when adding an opponent.
- [ ] Check range coherence after every public card.
- [ ] Fall back to blueprint when solve quality is poor.

Done when:

- [ ] the solver is stable on representative 3-way spots
- [ ] regressions are detectable automatically

## Phase 4: 4-6 Player Approximation Layer

Goal: support larger tables without exact multiway equilibrium.

Recommended approach:

- collapse opponents into coalitions
- solve only the key 2-3 player subgames
- prune weak branches early
- use a learned value network at the leaves

Checklist:

- [ ] Define coalition rules for 4, 5, and 6 players.
- [ ] Decide when to merge opponents.
- [ ] Add heuristic pre-filtering for weak ranges.
- [ ] Add subgame selection for critical branches.
- [ ] Route leaf evaluation through the value network.
- [ ] Keep exact solving only for small local subgames.
- [ ] Add fallback rules when the approximation looks bad.

Done when:

- [ ] 4-6 way spots return a usable strategy
- [ ] solve time stays bounded
- [ ] quality is better than heuristic play

## Phase 5: Value Network Data Pipeline

Goal: generate training data even without exact 6-way solves.

Sources:

- solved 2-way and 3-way subgames
- coalition-approximation rollouts
- depth-limited re-solving outputs
- sampled terminal EV labels

Checklist:

- [ ] Define input features for public state and all active ranges.
- [ ] Define output targets for multiway EV or bucket values.
- [ ] Add dataset generation from 2-way and 3-way solver runs.
- [ ] Add dataset generation from approximate 4-6 way runs.
- [ ] Add label versioning by solver method.
- [ ] Add train/validation split by spot type.

Done when:

- [ ] the dataset can be built without exact 6-way equilibrium labels
- [ ] the source of each target is traceable

## Phase 6: Value Network Training

Goal: train a practical leaf evaluator.

- [ ] Start with a small MLP.
- [ ] Train first on easy solved spots.
- [ ] Fine-tune on harder approximate spots.
- [ ] Use mixed precision.
- [ ] Keep checkpointing simple.
- [ ] Validate on held-out states.

Done when:

- [ ] the model predicts reasonable leaf EVs
- [ ] it improves runtime solve quality

## Phase 7: Runtime Integration

Goal: wire the new pieces into one path.

- [ ] Load the correct blueprint or approximate model per spot.
- [ ] Build depth-limited public subtrees at runtime.
- [ ] Warm start from stored strategy data.
- [ ] Query the value network at depth limits.
- [ ] Return a root action distribution.

Done when:

- [ ] runtime can handle HU, 3-way, and approximate 4-6 way spots
- [ ] the path is deterministic and debuggable

## Phase 8: Validation And Guardrails

Goal: catch bad approximations early.

- [ ] Add monotonicity checks.
- [ ] Add sampled best-response checks.
- [ ] Add range consistency checks.
- [ ] Add regression tests for representative spots.
- [ ] Add fallback to blueprint or heuristic play.

Done when:

- [ ] failures are visible
- [ ] unsafe outputs do not reach runtime

## Suggested Order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

## Definition Of Success

- 3-player postflop is supported with a real solver path.
- 4-6 player spots are supported with controlled approximations.
- The value network can be trained from approximate labels.
- The runtime solver can choose between exact and approximate modes.
