# Skill: Build a GPU-accelerated CFR solver for NLHE (real-time + offline)

This skill describes how to design and implement a CFR-based poker solver that actually benefits from GPU acceleration. It focuses on the “science” from the GPU-CFR papers: (1) turn irregular tree traversal into regular, batchable compute, (2) separate CPU orchestration from GPU throughput work, and (3) keep memory layout and reductions as the real performance driver.

---

## 0) Project-specific implementation rules

This repository is a typed Python implementation.

Current package and tooling choices:
- `numpy`, `scipy`, `numba` for numeric and performance-critical CPU work
- `torch` for the first GPU/value-network path
- `pokerkit` for poker rules/state reference support where helpful
- `treys` for early hand evaluation support
- `pytest`, `ruff`, `mypy`, `pytest-benchmark` for quality and measurement

Code style rules for this project:
- Use Python type hints everywhere practical
- Prefer `dataclass`, `Enum`/`StrEnum`, `Protocol`, `Literal`, `TypedDict`, and small typed APIs
- Keep `mypy` strict and keep new code passing it
- Favor immutable domain models first unless mutation is clearly needed for performance
- Hide third-party package details behind our own interfaces

Folder structure:
- Yes, use subfolders
- Do not keep everything in one flat `src/pokergpu/` directory
- Group code by domain so solver internals can grow cleanly

Recommended package layout:
- `src/pokergpu/core/`
  - cards, board, actions, state, rules
- `src/pokergpu/eval/`
  - hand evaluators, wrappers around `treys`, future batch eval APIs
- `src/pokergpu/cfr/`
  - regrets, strategies, traversal, variants
- `src/pokergpu/tree/`
  - public tree builder, node arrays, infoset indexing
- `src/pokergpu/abstraction/`
  - action abstraction, card buckets, ranges
- `src/pokergpu/gpu/`
  - batch builders, device logic, inference path
- `src/pokergpu/training/`
  - datasets, value-model training, export
- `src/pokergpu/bench/`
  - benchmark helpers and scenarios
- `src/pokergpu/io/`
  - config, serialization, artifact loading/saving

Working rule:
- Start simple, but place new modules in the right subfolder early
- Keep public APIs small and typed
- Optimize only after correctness, tests, and benchmarks exist

## 1) Outcome

You will be able to implement a solver that supports:

Real-time re-solving (seconds per decision)
- Build a depth-limited public tree from the current position
- Run many CFR iterations inside a time budget
- Use a GPU value model for leaf evaluation (batched inference)
- Return a mixed strategy for the current decision node

Offline solving (hours/days)
- Solve an abstracted game (action + card abstraction)
- Either:
  - use full-tree traversal with bulk-synchronous GPU kernels (levelized DP), or
  - use sampling CFR + GPU batching
- Export a blueprint strategy table or distill into a neural policy

---

## 2) Core principle: CFR is memory-bound and irregular

In practice, CFR performance is dominated by:
- Memory bandwidth (reading/writing regret tables and strategy sums)
- Data locality (contiguous arrays win, hash maps lose)
- Synchronization (atomics and contention kill speed)
- Branch divergence (GPU hates deep irregular branching)

So “GPU CFR” is mostly about:
- preprocessing structures so iterations are regular
- using batched reductions instead of scattered updates
- keeping GPU fed with large, uniform workloads

---

## 3) Choose your GPU acceleration mode

### Mode A: Real-time pipeline (recommended first)
Best for NLHE decision-time solving.

CPU does:
- build tree, pruning, abstraction bookkeeping
- forward/backward passes over the public tree (parallel CPU threads)
- regret updates (parallel over infosets)
- batching leaf requests for GPU

GPU does:
- batched leaf evaluation with a neural network (range + public features → EV)
- optionally batched per-level propagation kernels (only if your tree is sufficiently regular)

Why it’s fast:
- leaf value inference is massively parallel
- batching turns work into GEMMs (GPU sweet spot)
- CPU and GPU overlap via streams (pipeline)

### Mode B: Offline “matrix/levelized CFR” (highest raw throughput)
Best when your game is static enough to preprocess and store.

Precompute:
- topological levels or sequence-form structures
- sparse/dense adjacency for reach/value propagation
- dense regret/strategy arrays

Run each iteration as:
- forward reach propagation (by level)
- backward utility propagation (reverse level)
- regret matching / RM+ (elementwise + reductions)
- average strategy accumulation

Why it’s fast:
- predictable memory access
- large kernels over contiguous arrays
- minimal branching; no recursion

Tradeoff:
- more preprocessing and memory
- less flexible for dynamic pruning / changing action sets

---

## 4) Non-negotiables for NLHE feasibility

You cannot solve true 6-max NLHE without approximation.

You must implement:
1) Action abstraction
- discretize bet sizes per node
- enforce NLHE rules (min-raise, all-in, stack caps)

2) Card abstraction (bucketing / embeddings)
- map private hands (and sometimes hand+board) into buckets
- represent ranges as vectors over buckets

3) Depth-limited solving (for real-time)
- solve to a depth cap (end of street or capped horizon)
- evaluate leaves with a value function (neural net)

---

## 5) Data layout: how to make GPU possible

### 5.1 Public tree storage (flat arrays)
Represent the public tree as arrays, not pointers:
- node_type[node] ∈ {P0, P1, chance, terminal, leaf}
- first_child[node], child_count[node]
- infoset_id[node] (for player decision nodes)
- chance_prob[child] (for chance nodes)
- terminal_payoff[node] or payoff_index[node]

### 5.2 Infosets (dense offsets)
Store regret and strategy sum in contiguous blocks:
- infoset_offset[I]
- action_count[I]
- regret[infoset_offset[I] + a]
- strat_sum[infoset_offset[I] + a]

### 5.3 Ranges (bucket vectors)
Use ranges over buckets:
- range_p[player][bucket] float32
This is the object passed to value nets and leaf evaluation.

---

## 6) Algorithm choices

### CFR variant
Use one of:
- CFR+ (regret matching+)
- DCFR (discounting)
These converge much faster in practice than vanilla CFR.

### Traversal choice
- Full traversal: feasible only for smaller abstractions / depth-limited trees
- Sampling CFR (MCCFR / outcome sampling): for very large trees, but requires careful variance control and batching

---

## 7) How GPU acceleration is actually applied

### 7.1 GPU batched leaf evaluation (most important)
At depth limit, you need EVs to backprop.

Leaf input typically includes:
- public features: board, pot, stacks, action history encoding
- both players’ ranges (bucket vectors)
- optionally position/player-to-act encoding

Leaf output:
- EV for each player (or advantage vs baseline)

Implementation steps:
1) During the forward pass, collect leaf nodes and build feature tensors
2) Run GPU inference in large batches (thousands+)
3) Scatter results into a `leaf_value[node]` array
4) Backward pass consumes these values

Performance notes:
- use fp16/bf16 inference if stable
- keep regrets in fp32
- overlap CPU batching with GPU inference using streams

### 7.2 GPU levelized propagation (optional)
Only worth it if:
- your iteration can be expressed as level-by-level kernels
- per-level work is big enough to amortize kernel launch overhead
- you avoid heavy atomics

You run:
- reach propagation kernels per level
- value propagation kernels per reverse level

### 7.3 Regret matching kernels (optional)
RM/RM+ can run on GPU if:
- infosets are huge in count (millions)
- action_count is small and uniform-ish
- you can avoid contention (each infoset written by one thread/block)

Otherwise, CPU parallel RM+ is often competitive.

---

## 8) Synchronization strategy (how you avoid atomics)

Three patterns, in decreasing “GPU purity”:

1) Owner-writes (best)
- design so each infoset update is owned by one thread/block

2) Privatize + reduce (best general-purpose)
- each worker accumulates local deltas
- reduce at a barrier into the global arrays
- typical for “batch deals / batch trajectories” parallelization

3) Atomics (last resort)
- only for low-contention counters
- generally too slow for high-contention regret updates

---

## 9) Real-time solver pipeline (reference design)

Given current state S:
1) Build depth-limited public tree T (CPU)
2) Initialize regrets / strategy sums for this solve (warm-start from caches if available)
3) Until time budget expires:
   a) Forward pass: compute reach probabilities (CPU parallel)
   b) Collect leaf eval tasks
   c) GPU inference on leaf batch
   d) Backward pass: compute counterfactual values (CPU parallel)
   e) Regret update (CPU parallel over infosets; CFR+/DCFR)
   f) Average strategy update
   g) Apply pruning / cleanup
4) Return root strategy (average or current depending on variant)

Warm-start / caching:
- cache subtree strategies by public state signature
- reuse value net outputs for repeated leaf queries within the hand

---

## 10) Offline solver design (two viable paths)

### Path 1: Static abstraction + matrix/levelized GPU CFR
- preprocess adjacency/levels
- run bulk-synchronous iterations

### Path 2: Sampling CFR + batching + GPU inference
- sample trajectories / chance outcomes
- batch evaluation and delta accumulation
- reduce deltas into global regret tables

---

## 11) Why speedups can be “huge”

You get large speedups when:
- leaf evaluation dominates and is moved to GPU in large batches
- the core iteration becomes regular, levelized, and memory-coalesced
- you remove hash maps, recursion, and pointer chasing
- you reduce synchronization (privatize + reduce)

You do NOT get huge speedups when:
- GPU is starved (small batches, many tiny kernels)
- regret updates require heavy contention/atomics
- your abstraction causes massive random access patterns
- you rebuild structures too often (no reuse)

---

## 12) Build order (do this to actually finish)

1) Implement CFR+ on a tiny game (Kuhn/Leduc) using dense arrays
2) Implement public-tree flat array representation + parallel CPU traversal
3) Add depth-limited solving + leaf evaluator stub
4) Integrate GPU batched inference for leaf evaluation
5) Move to HUNL postflop re-solving
6) Add multiway approximations only after you have a strong heads-up core

---

## 13) Deliverables checklist

Must-have modules:
- Public tree builder (action abstraction + NLHE rules)
- Hand bucketer / range representation
- CFR+ / DCFR loop with parallel CPU traversal
- Leaf value network + GPU batched inference
- Pruning + caching + warm-start

Nice-to-have:
- Levelized GPU propagation kernels
- GPU regret matching kernels
- Distributed multi-GPU for offline solves

---

## 14) “Definition of done”

Real-time:
- Can re-solve a postflop position within 1–5 seconds
- Produces stable mixed strategies across repeated runs
- GPU utilization is high during inference (large batches)
- CPU threads are saturated during traversal/updates

Offline:
- Can run for hours without numerical blowups
- Exploitability decreases as iterations increase
- Strategy tables export cleanly and can be reloaded

---
