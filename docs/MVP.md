# MVP Plan

## Goal

Build a working heads-up postflop solver for no-limit Texas Hold'em that:
- uses the real evaluator
- respects dead cards and card removal
- builds a depth-limited public tree
- runs CFR+ on the abstracted tree
- returns root strategy, root EV, and action EVs

## MVP Scope

In scope:
- heads-up only
- postflop only
- fixed limited action set
- depth-limited public tree
- real hand evaluation
- real terminal EV calculation
- dead-card handling
- root strategy output
- root EV output
- per-action EV output
- warm start cache support

Out of scope:
- multiway equilibrium solving
- preflop solving
- learned value network training
- advanced bucketing optimization
- full-game exact NLHE solving

## Plan

### [x] 1. Lock the solver contract

Checklist:
- define one public solve input type for a postflop heads-up spot
- define one public solve output type
- include:
  - root strategy
  - root EV for both players
  - action EVs for each root action
  - iteration count
  - elapsed time
  - node count
  - leaf count
- define solver versioning for cache compatibility
- define deterministic seeds for repeatable runs

Done when:
- the runtime API is stable enough for tests and benchmarks

### 2. Define the restricted action set

Checklist:
- support only these action families for MVP:
  - check
  - fold
  - call
  - bet size a (fraction of pot)
  - bet size b (fraction of pot)
  - bet size c (fraction of pot)
  - all-in
- map each action family to concrete legal actions per node
- different action set per streets
- handel all-in correctly, not always it should be able (like do not all-in with low-equity or whatever)
- enforce min-raise, stack, and pot constraints
- reject illegal actions early
- keep action ordering deterministic

Done when:
- every legal node exposes a compact fixed action menu
- illegal actions never reach the solver

### 3. Build the public tree correctly

Checklist:
- build depth-limited public trees from the current game state
- stop expansion at the configured depth
- mark frontier nodes clearly
- preserve street transitions
- preserve turn order
- preserve terminal nodes
- preserve chance nodes for board runouts
- keep the tree flat-array friendly

Done when:
- a heads-up postflop state produces a valid public tree every time

### 4. Use the real evaluator

Checklist:
- replace heuristic leaf values with exact evaluator-backed values
- evaluate terminal and frontier states using actual hand strength logic
- include board texture and hole cards
- include dead-card removal
- handle board cards already on table
- handle future runout cards consistently
- support pot and stack context in EV calculation

Done when:
- leaf values come from real poker logic, not fixed stubs or fake bias terms

### 5. Implement dead-card and range handling

Checklist:
- remove impossible holdings after every board deal
- zero out blocked hands
- renormalize ranges after filtering
- keep range updates consistent across streets
- ensure canonical card handling matches the evaluator
- validate that no impossible combination survives into EV calculation

Done when:
- solver never assigns weight to hands blocked by the board

### 6. Compute EVs from the tree

Checklist:
- compute action EV for every root action
- compute root strategy from average strategy or current strategy, depending on output mode
- compute root EV from action EVs and strategy weights
- compute per-player EVs
- keep zero-sum consistency
- normalize EV to pot or chip units consistently

Done when:
- the solver returns a coherent root strategy and EV summary

### 7. Add CFR+ training loop

Checklist:
- use CFR+ regret clamping
- accumulate average strategy
- support repeated iterations
- keep regret and strategy storage dense
- keep traversal deterministic
- keep update rules compatible with depth-limited frontier evaluation
- verify convergence on tiny known games before trusting postflop results

Done when:
- the solver improves across iterations instead of drifting randomly

### 8. Make it GPU-friendly without fake values

Checklist:
- keep the evaluator real
- keep the GPU path for batch evaluation only
- remove any fixed-value GPU stub from the critical path
- use GPU for batched exact or model-backed leaf evaluation
- keep tree traversal and regret updates compatible with GPU batching
- preserve CPU fallback only as a development path, not as solver logic

Done when:
- GPU accelerates real leaf work, not placeholder math

### 9. Add cache and warm start

Checklist:
- key cache entries by public state fingerprint
- store warm-start regrets and strategy sums
- reload warm starts before solving
- blend cached state deterministically
- invalidate cache on solver version changes

Done when:
- repeated solves of the same spot get faster

### 10. Validate the solver

Checklist:
- test heads-up postflop on several boards
- test blocked-card handling
- test legal action generation
- test terminal EV calculation
- test CFR+ iteration behavior
- test output shape and determinism
- test warm-start reuse
- benchmark solve time and leaf throughput

Done when:
- the solver passes correctness tests and produces stable outputs on repeat runs

## MVP Acceptance Criteria

The MVP is done when all of these are true:
- heads-up postflop spots solve end to end
- action menu is limited and legal
- real evaluator is used
- dead cards are handled correctly
- depth limit works
- CFR+ runs
- root strategy is returned
- root EV is returned
- action EVs are returned
- results are deterministic for the same seed and state

## Next Phase After MVP

1. Add multiway approximation.
2. Add preflop blueprint solving.
3. Improve bucket quality.
4. Train a value network from solver output.
5. Optimize GPU throughput further.
