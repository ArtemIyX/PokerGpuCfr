# GPU-Accelerated CFR for No-Limit Hold’em (6-max) — Practical Build Guide

This document is a practical blueprint for building a fast CFR-based solver that can run either:
- **Offline** (hours/days, produce a “blueprint” strategy), or
- **Real-time** (seconds per decision, solve a subgame on the fly)

It focuses on *how* to use the GPU to get large speedups (often 10×–100× in the parts that are GPU-friendly), what *must* stay on CPU, and why the speedups happen.

Important reality check:
- **Full-game 6-max NLHE** is astronomically large. No existing system solves it “exactly” without abstraction and/or learning.
- The workable target is: **solve an abstracted game** offline, and/or **solve depth-limited subgames** in real time with learned leaf evaluation.

---

## Docs used

- [CFR-Explained](https://github.com/brianberns/CFR-Explained#parallelization)
- [A GPU implementation ofCounterfactual Regret Minimization](https://repositorio-aberto.up.pt/bitstream/10216/83517/2/35409.pdf#7#5)
- [Real-Time Parallel Counterfactual Regret Minimization](https://export.arxiv.org/pdf/2605.19928)
- [GPU-ACCELERATED COUNTERFACTUAL REGRET
MINIMIZATION](https://arxiv-org.ezproxy.obspm.fr/pdf/2408.14778v1)
## 1) What CFR is doing (the compute you must accelerate)

CFR iterates:
1. For each player, compute counterfactual values over the game tree under current strategies.
2. Update **regrets** at each information set.
3. Derive a new strategy (Regret Matching / Regret Matching+).
4. Accumulate an **average strategy** (or use variants that make current strategy good).

The expensive part is repeated *tree evaluation* + *regret updates* at huge scale.

Two high-level CFR families you’ll actually use:
- **CFR+ / DCFR** for faster practical convergence.
- **Sampling CFR** (MCCFR / outcome sampling) when full traversal is too expensive.

---

## 2) Where GPU helps (and where it doesn’t)

GPU helps when you can turn your workload into:
- large batches,
- regular memory access,
- lots of identical operations (SIMD-like),
- few branches,
- limited synchronization.

GPU struggles when you have:
- recursive traversal with branching,
- random accesses into giant hash maps,
- frequent contention/atomics on shared tables.

So you typically split the solver into CPU vs GPU responsibilities:

### Real-time solver (typical)
- CPU: build/update the public tree, pruning, abstraction bookkeeping, scheduling, reductions.
- GPU: **batched leaf evaluation** (neural value networks), and optionally batched/levelized DP passes.

### Offline solver (typical)
Two main GPU approaches:
1) **Batching + reduce** (CPU orchestration, GPU accelerates evaluation batches and/or parts of traversal)
2) **“CFR as linear algebra”**: rewrite the update into dense/sparse matrix-vector ops so GPU kernels run at full throughput (best speedups, but higher memory use and more preprocessing).

---

## 3) Requirements for any practical NLHE (6-max) solver

You need **abstraction**. Otherwise you won’t fit memory or compute.

### 3.1 Action abstraction
In NLHE, betting is continuous in theory. You must discretize:
- Example: {check, call, fold, bet 33%, bet 75%, bet 150%, all-in} per node,
- plus special-case constraints by stack depth, pot size, min-raise rules, etc.

Real-time solvers often re-solve with an action abstraction that depends on the state (SPR buckets, etc).

### 3.2 Information abstraction (cards)
You must bucket private hands:
- Preflop: precomputed clusters or abstractions by equity/realization.
- Postflop: buckets by hand strength, draw type, blockers, equity vs ranges, etc.
- Modern approach: use a learned embedding/value model and treat it as implicit abstraction.

### 3.3 Public tree and depth-limited solving
Real-time: you only solve from “now” until a depth limit (often to end of street or to river with a cap),
and then evaluate leaf nodes with a **value function** (neural net).

Offline: you can solve deeper but still will use abstraction + pruning.

---

## 4) Core data representation (make GPU possible)

To make GPU acceleration realistic, avoid maps and pointers at runtime.

### 4.1 “Public tree” nodes and indexing
Represent the public tree as flat arrays:
- `node_type[node_id]` (player/chance/terminal/leaf)
- `first_child[node_id]`, `child_count[node_id]`
- `infoset_id[node_id]` for player nodes
- `terminal_payoff[node_id]` for terminals (or indices into payoff tables)
- `chance_outcomes` stored as ranges of children with probabilities

### 4.2 Infosets and regret tables
For each infoset `I`:
- `A(I)` actions (small, e.g. 3–10)
- regrets `R[I, a]` and strategy sums `S[I, a]`

**GPU-friendly storage**:
- Use **dense arrays**: `regret[infoset_offset + a]`
- Precompute `infoset_offset[I]`, `action_count[I]`

Avoid atomics in the hot loop when possible:
- Prefer per-thread/per-block partial accumulators + reduction.

### 4.3 Card buckets
When you abstract private hands into buckets, the solver state per leaf becomes:
- (public node id, player range over buckets)

This dramatically shrinks GPU batch sizes:
- range vectors become `float32[NumBuckets]` per player.

---

## 5) Two GPU acceleration strategies that actually work

## Strategy A — Real-time “Parallel CFR pipeline” with GPU leaf evaluation

This is the most practical path for NLHE real-time play.

### A1) What you do on CPU
- Build a depth-limited public tree from the current game state.
- Apply:
  - action abstraction,
  - pruning (skip dominated/low-probability actions),
  - caching (reuse subtrees),
  - scheduling and parallelization across infosets/nodes.

### A2) What you offload to GPU
At each CFR iteration, you need leaf counterfactual values.
Instead of rolling out to terminal, evaluate leaves with a neural net:

- Input: (public features, pot, stacks, board, action history encoding, both players’ range vectors/bucket distributions)
- Output: expected values for each player (or advantage values)

Run these **in large batches** on GPU:
- Collect leaf evaluation requests during traversal.
- Run GPU inference in one or a few large kernels (or via TensorRT/cuDNN).
- Scatter the results back to CPU buffers.

### A3) Why this gives big speedups
- Leaf evaluation is expensive and massively parallel.
- Batching turns small per-leaf work into large matrix multiplies (GPU sweet spot).
- CPU and GPU can overlap: while GPU runs inference for batch k, CPU prepares batch k+1.

### A4) Skeleton of a real-time iteration (pipeline)
One CFR iteration (conceptual):
1. Compute reach probabilities / weights forward through the tree.
2. Identify leaves needing evaluation (depth limit).
3. Batch leaf features → GPU inference.
4. Backprop utilities / counterfactual values.
5. Update regrets (CFR+/DCFR).
6. Update average strategy.
7. Apply pruning / cleanup.

Parallel dimensions:
- by infoset (many independent RM/RM+ ops),
- by node (many forward/backward computations).

### A5) Practical notes for NLHE “6-max”
Real-time multi-player CFR is harder (more players, bigger infosets).
Common workaround:
- Use **nested subgame solving** with approximations,
- Solve effectively 2-player subgames (or treat opponents as a coalition),
- Or use learned policies for off-tree agents.

If you insist on 6-max exactness, expect much heavier abstraction and weaker guarantees.

---

## Strategy B — Offline “CFR as matrix ops” (maximum GPU utilization)

This is the most “100×-ish” approach in pure solver throughput, but costs more memory and preprocessing.

### B1) Main idea
Rewrite each CFR iteration into:
- sparse/dense matrix-vector operations,
- reductions,
- elementwise transforms.

This removes recursion and irregular branching.
The solver becomes bulk synchronous:
- forward pass (reach propagation),
- backward pass (utility propagation),
- regret update pass (RM/RM+),
- average strategy accumulation.

### B2) How to get there (levelization)
Preprocess the game graph:
- Topologically order nodes or group them by depth (“levels”).
- For each level, precompute which nodes read which parents and write which children.
- Store adjacency in CSR/COO format (GPU sparse-friendly).

### B3) Memory layout
You want contiguous arrays:
- `pi[node]` reach probs (or reach per player)
- `v[node]` values
- `cfv[node]` counterfactual values
- infoset strategy `sigma[I,a]`
- regrets `R[I,a]`

If you use buckets:
- treat each (node, bucket) as a “state row”
- then operations become block-sparse.

### B4) Where the speed comes from
- GPU excels at SpMV/GEMV-like workloads with high arithmetic intensity.
- No recursion, minimal branch divergence.
- Coalesced reads/writes.
- Reductions are done via optimized primitives.

### B5) Tradeoffs
- Much higher memory usage (you store more structure explicitly).
- More preprocessing time (build sparse structures).
- Less flexible for dynamic pruning/action sets unless you rebuild structures.

---

## 6) Picking your architecture: real-time vs offline

### Real-time recommended architecture (most people should build this)
- CFR variant: CFR+ or DCFR
- Depth-limited public tree for current street
- Action abstraction (small set)
- Hand abstraction (buckets or learned embeddings)
- GPU: neural leaf evaluation (batched), optional batched value propagation
- CPU threads: traversal, regret updates, pruning, scheduling
- Output: local strategy for the current decision

### Offline recommended architecture
Two options:
1) If your abstraction is not gigantic:
   - Matrix-op CFR on GPU (Strategy B)
2) If your abstraction is huge and dynamic:
   - Sampling CFR + batching (GPU used for inference/evaluation and batch reductions)

Output: blueprint strategy tables or a learned policy/value model.

---

## 7) Implementation details that determine whether you get 10× or 100×

### 7.1 Stop doing random hash-map accesses in the hot loop
If regrets live in a `unordered_map<infoset, vector>` you will not get GPU speed.
Use:
- perfect hashing or indexing,
- contiguous arrays with offsets,
- compact action lists.

### 7.2 Avoid global atomics on regret tables
If many threads update the same infoset:
- Either (a) give each block a private accumulator and reduce,
- Or (b) batch trajectories and reduce on CPU,
- Or (c) design traversal so each thread owns disjoint infosets (often possible in levelized passes).

### 7.3 Batch everything
GPU wins only if you keep it fed:
- batch leaf eval requests (thousands+),
- batch infoset RM+ updates (millions+ small ops),
- batch sparse ops per level.

### 7.4 Overlap CPU and GPU
Use CUDA streams (or equivalent):
- Stream 0: leaf inference batch k
- Stream 1: leaf inference batch k+1
- CPU threads: build next batch while GPU runs

### 7.5 Use the right numeric precision
- Inference: fp16/bf16 often fine.
- Regrets: fp32 is safer (numerical drift matters over many iterations).
- Try mixed precision: store regrets fp32, store reach/value fp16 where stable.

### 7.6 Keep action counts tiny
CFR runtime is roughly proportional to total (infosets × actions).
In NLHE, action abstraction quality matters more than “more bet sizes”.

---

## 8) A concrete “real-time postflop solver” recipe

### Step 1: Define abstractions
- Action set per node: choose 5–9 actions.
- Buckets: 500–3000 postflop buckets (start small).
- Range representation: `float32[NumBuckets]`.

### Step 2: Build the public tree (from current state)
- Expand actions until:
  - street ends, or
  - depth limit reached, or
  - node count cap reached.

Mark depth-limit nodes as leaf-evaluated.

### Step 3: Leaf value network
Train a value model:
- Inputs: board features, pot/stacks, public history encoding, both players’ ranges
- Outputs: EV for each player (or advantage)

In runtime:
- gather leaf inputs → run GPU inference in large batches.

### Step 4: Parallel CFR iteration loop (2–8 seconds budget)
Repeat until time budget:
1) forward reach pass (CPU parallel over nodes)
2) collect leaves and run GPU eval
3) backward utility pass (CPU parallel over nodes)
4) regret update (CPU parallel over infosets; RM+)
5) average strategy update
6) apply pruning heuristics

Return strategy at root infoset.

### Step 5: Practical pruning
- Remove actions with consistently negative regret beyond a threshold (careful with guarantees).
- Skip low-probability branches (reach below epsilon).
- Cache subtrees across similar states (if you do multiple decisions in same hand).

---

## 9) A concrete “offline abstracted 6-max” recipe (feasible version)

True 6-max equilibrium is out of reach without heavy approximation.
A feasible target:
- Solve *street-by-street* abstractions,
- Train blueprint policies/value functions,
- Use real-time re-solving only in critical spots (often heads-up or 2–3-way).

Offline loop:
1) Generate an abstract game (action + card abstraction).
2) Use CFR+/DCFR:
   - either full traversal (if small enough),
   - or sampling CFR (if huge).
3) Use GPU for:
   - batch evaluation / learned value targets,
   - matrix-op CFR if your abstract game is static and fits memory.
4) Export blueprint strategy tables or distill into a neural policy.

---

## 10) Why “100× faster” can be real (and when it’s not)

You can see huge speedups when:
- your baseline is single-threaded CPU,
- most of your cost is in leaf evaluation and/or regularized passes,
- you use dense arrays and large batches,
- you avoid atomics and random memory access.

You will NOT see 100× when:
- the problem is dominated by dynamic branching + hash lookups,
- the GPU is starved (tiny batches),
- you’re rebuilding structures too often (no reuse),
- your action abstraction is huge (too many actions per infoset).

In practice:
- Real-time solvers often get a few × speedup per iteration plus a big leaf-eval gain.
- Offline matrix-op CFR can get the largest raw throughput gains if memory fits.

---

## 11) Minimal pseudocode (real-time pipeline)

Given public tree `T`, infosets `I`, regrets `R`, strat sums `S`, value net `VNet`:

for iter in 1..until_time_budget:
  # forward pass
  compute_reach_probs_parallel(T, current_sigma)

  # leaf batch
  leaves = collect_leaf_nodes(T)
  batch = build_leaf_features(leaves, ranges, public_features)
  leaf_values = GPU_infer(VNet, batch)   # batched inference
  write_leaf_values(leaves, leaf_values)

  # backward pass
  compute_counterfactual_values_parallel(T)

  # regret updates
  for infoset in parallel(I):
    sigma = regret_matching_plus(R[infoset])
    R[infoset] += instantaneous_regret(infoset, sigma, cf_values)
    S[infoset] += weight * sigma

return average_strategy_at_root(S)  # or current strategy in CFR+

---

## 12) What to build first (pragmatic progression)

1) Implement a tiny game (Kuhn/Leduc) with CFR+ on CPU using dense arrays.
2) Add batching + parallel CPU across infosets.
3) Add a GPU leaf-eval stub (even a fake network) to validate batching/integration.
4) Move to HUNL postflop depth-limited re-solving.
5) Only then attempt multiway (6-max) approximations.

---

## 13) Checklist: the non-negotiables for GPU performance

- Flat arrays, not pointer graphs.
- Precomputed offsets for infoset/action storage.
- Large batches (thousands+) for GPU inference.
- Minimal divergence in GPU kernels.
- Avoid atomics on shared regret tables in inner loops.
- Overlap CPU prep with GPU execution (streams).

---