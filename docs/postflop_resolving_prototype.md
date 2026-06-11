# Postflop Re-Solving Prototype

Goal: add the first runtime postflop re-solving path for heads-up spots.

This is the spec for task 14 in `PLAN.md`.

## Scope

- Heads-up only.
- Postflop only.
- One public subtree per solve.
- Time-budgeted local solve.
- Return the mixed strategy at the root infoset.
- Provide one CLI or script demo.

## Non-goals

- Multiway solving.
- Full-game equilibrium.
- Turn/river blueprint training.
- Advanced abstraction search.
- Persistent caching.

## Why this exists

The project already has:

- flat public trees
- infoset stores
- CFR iteration logic
- leaf evaluation plumbing

This prototype connects those pieces into a runtime re-solver for one postflop decision.

## Target user flow

1. Load a real game state.
2. Build the public subtree for the current spot.
3. Attach player ranges at the root.
4. Run CFR-style iterations until the time budget expires.
5. Read the root average strategy.
6. Use that strategy for the next action.

## Current code concepts to reuse

- `pokergpu.tree.PublicTree`
- `pokergpu.tree.builder.build_public_tree`
- `pokergpu.tree.builder.TreeBuildConfig`
- `pokergpu.cfr.InfosetStore`
- `pokergpu.cfr.iteration`
- `pokergpu.eval.LeafFeature`
- `pokergpu.eval.LeafFeatureBatch`
- `pokergpu.eval.LeafValueBatch`

## Prototype assumptions

- Two active players only.
- Root state is already postflop.
- Board cards are known.
- Each player has a range vector over private hands or buckets.
- Leaf evaluation can be CPU stub first, GPU later.
- Action abstraction can stay simple and local to the subtree.

## Required outputs

The resolver should return:

- root infoset id
- mixed strategy for each root action
- optional raw regrets and strategy sums for debugging
- solve metadata: iterations, elapsed time, node count, leaf count

## Public subtree builder

Build a subtree from the current public state.

Minimum requirements:

- root is the current game state
- tree contains only legal actions from that state
- nodes keep the existing flat-array layout
- frontier nodes are marked for leaf evaluation
- node metadata records which state each node represents

Preferred behavior:

- limit by depth and/or node count
- stop expansion at terminal states
- stop expansion at showdown or leaf cutoff
- keep children contiguous in memory

Questions the builder must answer:

- Which player acts at each node?
- Which infoset does each player node map to?
- Which frontier nodes need value estimates?
- Which action sequence led to each node?

## Root ranges

The root must accept player ranges.

Minimum format:

- `range_p0`
- `range_p1`

Each range should:

- sum to 1 before card removal
- be masked for board card conflicts
- be normalized after masking

If the project uses buckets at runtime, the resolver must accept bucket-weight vectors instead of raw hand combos.

Required range handling:

- apply board card removal at root
- preserve only legal private holdings
- renormalize after masking
- reject impossible zero-mass ranges

## Leaf evaluation

Leaf nodes need EV estimates.

Prototype order:

1. CPU stub evaluator.
2. GPU batch evaluator.
3. Learned value model later.

Leaf features should include at least:

- node index
- player to act
- street
- pot
- stacks
- board size
- player reaches
- terminal flag
- frontier flag
- infoset id

Leaf values should return:

- EV for player 0
- EV for player 1

## Solving loop

Use a time-budgeted loop.

Skeleton:

1. Build or warm-start solver state.
2. Traverse the subtree forward.
3. Evaluate frontier leaves in batch.
4. Backpropagate values.
5. Update regrets.
6. Update average strategy sums.
7. Repeat until deadline.

Recommended stopping criteria:

- wall-clock budget reached
- iteration cap reached
- no new frontier nodes
- invalid state detected

## Strategy update rule

Use the existing infoset store pattern:

- regret matching for current strategy
- regret accumulation per infoset
- strategy-sum accumulation for final output

For the prototype, CFR+ style non-negative regrets are acceptable if that matches the current solver core.

## Warm start

Optional but useful.

If blueprint data exists later, warm start from it. For now, start from:

- uniform strategy
- zero regrets
- normalized root ranges

Do not block the prototype on blueprint availability.

## Demo scenario

Provide one deterministic postflop spot, for example:

- heads-up flop
- fixed stacks
- fixed pot
- fixed board
- fixed action history
- fixed root ranges

The demo should print:

- board and pot
- root action list
- root mixed strategy
- solve time
- iteration count

## CLI or script shape

Preferred command style:

```bash
python -m pokergpu.cli postflop-resolve --spot <name>
```

Or a script under `scripts/` if CLI integration is not ready.

The command should:

- build the subtree
- run the solver
- print the root strategy
- exit cleanly

## Validation checks

Add checks for:

- tree node count and child bounds
- legal root ranges
- masked ranges summing to 1
- non-empty root actions
- strategy sums normalizing correctly
- deterministic demo output on the same seed

## Acceptance criteria

The task is done when:

- a heads-up postflop subtree can be built from a game state
- root player ranges are accepted and masked correctly
- the solve loop runs for a fixed time budget
- the root mixed strategy is returned
- one demo spot works from CLI or script
- tests cover the main data-path assumptions

## Suggested implementation order

1. Define the runtime spot input type.
2. Build subtree construction from `GameState`.
3. Add range masking at root.
4. Wire leaf batching into the traversal loop.
5. Add timed CFR iteration wrapper.
6. Return root mixed strategy.
7. Add one demo entrypoint.
8. Add tests.

## Open design points

- exact root input schema
- hand-range vs bucket-range representation
- how deep the first subtree should go
- whether to reuse the existing traversal code or add a runtime-specific wrapper
- whether the first demo uses CPU leaf eval only

## Notes for later phases

This prototype should be the smallest useful runtime resolver.

After this lands, the next steps are usually:

- warm start from blueprint
- better postflop abstraction
- cached subtrees
- GPU leaf inference
- stronger demo spot coverage
