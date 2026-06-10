# GPU-Accelerated CFR for No-Limit Hold'em (6-max) — Practical Build Guide

This document covers building a GPU-accelerated CFR solver for 6-max NLHE in two modes:
- **Offline / Research**: generate blueprint strategies (hours/days on GPU+CPU)
- **Runtime / Real-game**: real-time subgame re-solving during play (seconds per decision, GPU-primary)

The architecture follows the Libratus/Pluribus pattern: offline blueprint + online depth-limited re-solving.

---

## Reference Repos

- [GPU-accelerated GTO poker solver](https://github.com/a9876543245/DEEPFOLD-SOLVER)
- [GPU CFR](https://github.com/janrvdolf/gpucfr)
- [Counterfactual Regret Minimization in JAX](https://github.com/Egiob/cfrx)
- [CFR-Explained](https://github.com/brianberns/CFR-Explained#parallelization)

## Reference Docs

- [A GPU implementation of CFR](https://repositorio-aberto.up.pt/bitstream/10216/83517/2/35409.pdf)
- [Real-Time Parallel CFR](https://export.arxiv.org/pdf/2605.19928)
- [GPU-Accelerated CFR](https://arxiv-org.ezproxy.obspm.fr/pdf/2408.14778v1)

---

## 1. System Architecture Overview

The full system has three layers:

```
[OFFLINE LAYER]
  Abstract game construction (card + action abstraction)
      ↓
  GPU/CPU CFR training (CFR+/DCFR/MCCFR on abstracted tree)
      ↓
  Blueprint strategy tables (regrets/avg strategy per infoset)
      ↓
  Optional but not MVP: distill blueprint into neural policy/value network

[RUNTIME LAYER]
  Game state arrives (street + ranges + pot + stacks)
      ↓
  Build depth-limited public subtree
      ↓
  GPU-accelerated subgame re-solving (warm-started from blueprint)
      ↓
  Return action distribution at root infoset

[VALUE NETWORK LAYER]  (supports both)
  Leaf node evaluation model
  Trained offline via solver rollouts
  Used at depth limit in runtime; used as bootstrap in offline
```

---

## 2. What CFR Is Doing (Compute to Accelerate)

CFR iterates:
1. For each player, compute counterfactual values over the game tree under current strategies.
2. Update regrets at each information set.
3. Derive a new strategy (Regret Matching / Regret Matching+).
4. Accumulate an average strategy.

The expensive part: repeated tree evaluation + regret updates at huge scale.

CFR variants by use case:
- **CFR+ / DCFR**: offline blueprint training, faster practical convergence
- **MCCFR (outcome / external sampling)**: offline when full traversal is too expensive
- **DREAM (Deep Regret-based Abstraction and Minimization)**: GPU-native; better than MCCFR for learned-value pipelines; generates training data for the value network simultaneously

---

## 3. Abstraction (Required for 6-max NLHE)

Full 6-max NLHE is unsolvable without abstraction. You need both card and action abstraction.

### 3.1 Action Abstraction

Discretize betting into a small fixed set per node:
- Example: `{fold, check/call, bet 33%, bet 75%, bet 150%, all-in}`
- Vary by street (preflop tighter, postflop richer)
- Constraint by SPR, pot size, min-raise rules

Action count directly controls solver cost: fewer actions = faster convergence. Quality matters more than quantity.

### 3.2 Card / Information Abstraction

**Preflop**: 169 canonical hands (suit isomorphism). No bucketing needed at this scale.

**Postflop bucketing** (critical for GPU feasibility):
- Target: 500–5000 buckets per street
- Cluster hands by: equity vs. random range, equity vs. opponent range, draw potential, blocker value
- Use **Earth Mover's Distance (EMD)** to measure bucket similarity (standard, produces better clusters than L2)
- Use **potential-aware abstraction**: bucket by (current equity, equity on future streets) pair — preserves draw value
- Tools: k-means on equity histograms, or learned embeddings from a neural encoder

Range representation per node: `float32[NumBuckets]` per player. This is what the solver actually propagates.

### 3.3 Multiway Approximation

True 6-max equilibrium is not computable. Practical approaches:
- Treat 6-max as a sequence of 2-player and 3-player subgames
- Use **coalition approximation**: treat remaining players as a single agent with aggregated range
- Solve heads-up subgames exactly; approximate multiway nodes with heuristics or learned values
- Pluribus used this: blueprint for all streets, real-time re-solving only in critical spots (often 2–3-way)

---

## 4. Core Data Representation

Avoid pointer graphs and hash maps at runtime. Use flat arrays throughout.

### 4.1 Public Tree Nodes

```
node_type[node_id]         // PLAYER / CHANCE / TERMINAL / LEAF
first_child[node_id]
child_count[node_id]
action_id[node_id]         // which action led to this node
infoset_id[node_id]        // for PLAYER nodes
terminal_payoff[node_id]   // for TERMINAL nodes (indexed into payoff table)
chance_prob[node_id]       // for CHANCE nodes
depth[node_id]
street[node_id]
```

### 4.2 Infoset and Regret Tables

For each infoset `I` with `|A(I)|` actions:

```
regret[infoset_offset[I] + a]      // float32, CFR+ clamped to 0
strat_sum[infoset_offset[I] + a]   // float32, for average strategy
current_sigma[infoset_offset[I] + a]
```

`infoset_offset[I]` precomputed. Dense contiguous storage. No hash maps in the hot loop.

Regrets stay float32 — numerical drift matters over millions of iterations. Reach/value can be float16 where stable.

### 4.3 Range Vectors

Per (node, player): `float32[NumBuckets]`

Normalize after chance events. For multiway: one range vector per active player.

### 4.4 Blueprint Tables

After offline training, export:
```
blueprint_sigma[infoset_id][action]    // average strategy, compressed
blueprint_regret[infoset_id][action]   // optional: warm-start re-solving
```

Store compressed (quantized to uint16 or uint8 per action with normalization). At runtime, load relevant subtree into VRAM before re-solving begins.

---

## 5. Offline Mode — Blueprint Generation

### 5.1 Goal

Produce a strategy table over the full abstract game. Used directly in low-variance spots and as warm-start for runtime re-solving.

### 5.2 CFR Variant Choice

| Abstraction size | Recommended variant |
|---|---|
| Fits in VRAM (< 12GB) | Matrix-op CFR+ on GPU (Strategy B below) |
| Larger, sampling required | MCCFR (external sampling) + GPU batch eval |
| Learned value function available | DREAM — GPU-native, generates value training data |

### 5.3 Strategy A — MCCFR + GPU Batch Evaluation

CPU orchestrates tree traversal (sampling). GPU accelerates:
- Leaf node evaluation (neural value net, large batches)
- Batch regret accumulation after sampling episodes

Parallelism: run many independent MCCFR trajectories in parallel CPU threads. Collect leaf eval requests, dispatch to GPU in batches of 4096+.

### 5.4 Strategy B — Matrix-Op CFR (Maximum GPU Throughput)

Rewrite CFR iteration as bulk-synchronous linear algebra:

**Preprocessing (once)**:
- Topologically sort nodes by depth (levelization)
- Build CSR/COO adjacency for forward/backward passes
- Precompute reach contribution matrices per level

**Per iteration (GPU)**:
1. Forward pass: propagate reach probabilities level by level (SpMV per level)
2. Leaf evaluation: GPU neural inference (if depth-limited) or terminal payoff lookup
3. Backward pass: propagate counterfactual values level by level (SpMV transpose)
4. Regret update pass: elementwise RM+ per infoset (parallelized over all infosets)
5. Average strategy accumulation

```python
# Pseudocode (JAX/PyTorch style)
for level in levels_bottom_up:
    nodes = level_nodes[level]
    # backward: cfv[parent] += sigma[I,a] * cfv[child] * reach_opponent
    cfv = sparse_backward(adjacency[level], cfv_children, sigma, reach)

for infoset in all_infosets:  # parallelized
    r = instant_regret(infoset, cfv, reach)
    R[infoset] = jnp.maximum(R[infoset] + r, 0)  # CFR+
    S[infoset] += t_weight * sigma_from_regret(R[infoset])
```

**Memory constraints**: on 16GB VRAM (RTX 5080), this limits abstract game size. Budget:
- Regret table: `num_infosets * max_actions * 4 bytes`
- Reach/value arrays: `num_nodes * num_buckets * 4 bytes`
- If this exceeds VRAM, fall back to MCCFR + GPU eval (Strategy A)

### 5.5 DREAM for Offline Training

DREAM (Deep Regret-based Abstraction and Minimization) is GPU-native and preferred when you also want to train a value network:

- Maintains a **reservoir buffer** of (state, regret) samples
- Trains a neural net to predict regrets from game state features
- CFR traversal is driven by the learned regret model, not explicit tables
- Generates value network training data as a byproduct of training

Advantage over MCCFR: no explicit regret table needed (neural approximation); scales better to large state spaces; GPU utilization is higher (training dominates compute).

### 5.6 Offline Training Pipeline

```
1. Build abstract game (action + card abstraction)
2. Initialize regret/strategy tables to zero (or uniform)
3. Run CFR+/DCFR/DREAM for N iterations:
   a. GPU: batch leaf eval / matrix passes
   b. CPU: sampling, scheduling, pruning (for MCCFR)
4. Every K iterations:
   a. Checkpoint average strategy
   b. Compute exploitability estimate (sampled)
   c. Log convergence
5. Export blueprint tables
6. Optional: distill blueprint into neural policy (behavior cloning + fine-tuning)
```

### 5.7 Parallelism Across CPU and GPU (Offline)

- Use CUDA streams: stream 0 runs current batch, stream 1 prepares next
- CPU threads handle: sampling new trajectories, pruning, range normalization, scheduling
- Overlap CPU prep with GPU kernel execution at all times
- For MCCFR: multiple CPU threads each run independent traversals, coalesce leaf eval requests into GPU batches

---

## 6. Runtime Mode — Real-Time Re-Solving

### 6.1 Goal

At each decision point in a live game, solve the current subgame to near-Nash within the time budget (2–8 seconds). Warm-start from blueprint.

### 6.2 What Stays on CPU

- Build depth-limited public tree from current state
- Apply action abstraction (same set as offline, for blueprint compatibility)
- Pruning: skip branches with reach below epsilon
- Regret updates (RM+) — fast enough on CPU for depth-limited trees
- Scheduling and synchronization

### 6.3 What Goes to GPU

- **Leaf node evaluation**: neural value network, batched
- **Batched RM+ updates**: if infoset count is large (millions), GPU parallelism helps
- **Range propagation**: forward/backward passes if tree is large enough to justify transfer overhead

### 6.4 Warm-Starting from Blueprint

Before beginning re-solving:
1. Load blueprint regrets/strategy for all infosets in the current subtree into CPU memory (or VRAM if budget allows)
2. Initialize `R[I,a]` from blueprint regrets (CFR+ warm-start)
3. Initialize `S[I,a]` from blueprint average strategy
4. Run re-solving iterations until time budget

This typically converges 10–50× faster than cold-start.

### 6.5 Real-Time Pipeline (Per Decision)

```
game state arrives
    ↓
build public tree (CPU, ~ms)
    ↓
load blueprint warm-start for this subtree
    ↓
for iter in 1..time_budget:
    1. forward reach pass      (CPU parallel over nodes)
    2. collect leaf nodes
    3. batch leaf features → GPU inference (VNet)
    4. scatter leaf values back
    5. backward cfv pass       (CPU parallel over nodes)
    6. regret update RM+       (CPU parallel over infosets)
    7. average strategy update
    8. prune low-reach branches
    ↓
return average_strategy[root_infoset]
```

### 6.6 GPU Overlap in Runtime

```
iteration k:
  CPU: build leaf feature batch k+1
  GPU stream 0: run VNet on batch k
  CPU: regret updates from batch k-1 results

iteration k+1:
  CPU: build leaf feature batch k+2
  GPU stream 0: run VNet on batch k+1
  CPU: regret updates from batch k results
```

Keep GPU fed at all times. Don't wait for GPU synchronously inside the CFR loop.

### 6.7 Multiway Handling at Runtime

6-max real-time:
- Identify players still in the hand with non-trivial ranges
- If 4+ players: use coalition approximation (treat 2+ opponents as one) + learned value correction
- If 3-way: solve 3-player subgame with approximations
- If heads-up: exact re-solving against single opponent range

The value network handles multiway implicitly if trained on multiway states.

---

## 7. Value Network

### 7.1 Role

Evaluates leaf nodes at depth limit. Takes public state + both players' range vectors as input, outputs EV per player (or advantage).

Replaces full rollout. Quality of the value network determines quality of runtime re-solving.

### 7.2 Architecture

Input features per leaf node:
- Board cards (one-hot or suit-normalized encoding)
- Pot size, stack sizes (normalized)
- Street
- Action history encoding (fixed-width sequence)
- Player 0 range: `float32[NumBuckets]`
- Player 1 range: `float32[NumBuckets]`
- (Multiway: one range per player)

Output:
- `EV[player]` for each active player (sum to pot, verify)
- Or: advantage value `A[player, bucket]` (richer, more expensive)

Architecture: MLP with 4–8 layers, residual connections, LayerNorm. Wider is better given GPU budget.

### 7.3 Training Data Generation

Option A (solver rollouts):
- Run offline CFR to convergence on many board/range samples
- Record (state, both ranges) → (EV per player) pairs
- ~10M–100M samples needed for good coverage

Option B (self-play + DREAM):
- DREAM generates (state, regret) samples during training
- Simultaneously train a value head on these samples

Option C (bootstrap from blueprint):
- Use blueprint strategy to simulate rollouts forward
- Cheap but biased toward blueprint quality

### 7.4 Deployment

- Export to TensorRT (fp16, optimized for batch inference)
- Warm VRAM cache at game start
- Batch size ≥ 1024 for full GPU utilization
- Typical latency: < 5ms per batch at 4096 leaves (RTX 5080)

---

## 8. Performance: Where 10× vs 100× Comes From

### You get large speedups when:
- Baseline is single-threaded CPU
- Most cost is in leaf evaluation and/or regular passes
- Dense arrays + large batches throughout
- No atomics or hash lookups in the hot loop
- GPU is kept fed (overlap CPU prep with GPU exec)

### You get little speedup when:
- Dynamic branching + hash map accesses dominate
- Batch sizes are small (GPU underutilized)
- Structures are rebuilt every iteration
- Action abstraction is too large (thousands of infosets × actions exhausts VRAM)

### Practical numbers (RTX 5080, 16GB VRAM):
- Matrix-op CFR (small abstract game, fits VRAM): 50–100× vs single-thread CPU
- MCCFR + GPU leaf eval (large game): 5–20× depending on batch efficiency
- Real-time re-solving (depth-limited, warm-start): 10–30× vs CPU-only, enabling 100+ CFR iterations in 5 seconds

---

## 9. Non-Negotiables for GPU Performance

- Flat arrays, not pointer graphs or hash maps
- Precomputed offsets for infoset/action storage
- Large batches (≥ 1024, ideally 4096+) for GPU inference
- Minimal branch divergence in GPU kernels
- No global atomics on shared regret tables in inner loops (use per-block accumulators + reduction)
- Overlap CPU prep with GPU execution via CUDA streams
- Float32 for regrets; float16/bfloat16 for reach/value where numerically stable
- Keep action count small (5–9 per node); action abstraction quality over quantity
- Blueprint warm-start in runtime mode (non-negotiable for convergence speed)

---

## 10. Implementation Checklist (Pragmatic Progression)

**Phase 1 — Foundation**
- [ ] Implement CFR+ on Kuhn poker with dense flat arrays (CPU)
- [ ] Add RM+ and verify convergence to Nash
- [ ] Extend to Leduc Hold'em
- [ ] Measure exploitability per iteration

**Phase 2 — GPU Integration**
- [ ] Port regret update pass to GPU (CUDA/JAX/PyTorch)
- [ ] Add a stub value network (random, for pipeline validation)
- [ ] Implement CUDA stream overlap
- [ ] Benchmark: CPU baseline vs GPU version

**Phase 3 — Card Abstraction**
- [ ] Implement equity histogram computation (Monte Carlo or enumeration)
- [ ] K-means clustering with EMD distance for postflop buckets
- [ ] Validate bucket quality (equity preservation check)
- [ ] Add potential-aware abstraction for draw-heavy boards

**Phase 4 — HUNL Postflop Solver**
- [ ] Build depth-limited public tree generator
- [ ] Implement real-time pipeline (sections 6.4–6.6)
- [ ] Train first value network on solver rollout data
- [ ] Benchmark convergence quality vs time budget

**Phase 5 — Blueprint Generation**
- [ ] Implement full DCFR or MCCFR offline loop
- [ ] Add blueprint table serialization / compression
- [ ] Implement warm-start initialization in runtime solver
- [ ] Validate: re-solving with warm-start vs cold-start convergence speed

**Phase 6 — 6-max Multiway**
- [ ] Implement coalition approximation for 4–6 players
- [ ] Add multiway range propagation
- [ ] Train value network on multiway states
- [ ] Benchmark multiway re-solving quality

---

## 11. Memory Budget Reference (RTX 5080, 16GB VRAM)

| Structure | Formula | Example (1M infosets, 7 actions, 1000 buckets) |
|---|---|---|
| Regret table | `num_infosets × max_actions × 4B` | 28 MB |
| Strategy sum table | same | 28 MB |
| Reach arrays | `num_nodes × num_buckets × 4B` | ~4 GB (1M nodes) |
| Value arrays | same | ~4 GB |
| VNet weights | architecture-dependent | ~200 MB (medium MLP) |
| Adjacency (CSR) | `num_edges × 8B` | ~80 MB (10M edges) |

Keep total under 14GB to leave headroom for driver/kernel overhead. If reach/value arrays don't fit, tile by subtree or switch to MCCFR.

---

## 12. Full Pseudocode

### Offline (Matrix-Op CFR+)

```python
# Preprocessing
levels = topological_levels(tree)
adjacency = build_csr(tree)
infoset_offsets = precompute_offsets(infosets)

R = zeros(total_actions)       # regrets
S = zeros(total_actions)       # strategy sums
sigma = uniform(total_actions)

for t in 1..num_iterations:
    weight = compute_dcfr_weight(t)

    # Forward: reach probabilities
    pi = ones(num_nodes * num_buckets)
    for level in levels_top_down:
        pi = forward_pass(adjacency[level], pi, sigma)

    # Leaf evaluation (GPU)
    leaf_features = build_leaf_features(leaves, pi)
    leaf_ev = GPU_vnet_infer(leaf_features)  # batched, fp16

    # Backward: counterfactual values
    cfv = zeros(num_nodes * num_buckets)
    cfv[leaves] = leaf_ev
    for level in levels_bottom_up:
        cfv = backward_pass(adjacency[level], cfv, sigma, pi)

    # Regret update (GPU, parallelized over infosets)
    instant_r = compute_instant_regret(cfv, sigma, pi, infoset_offsets)
    R = maximum(R + instant_r, 0)          # CFR+ floor
    sigma = regret_matching(R, infoset_offsets)
    S += weight * sigma                     # DCFR weighted average

blueprint = normalize_strategy(S, infoset_offsets)
```

### Runtime (Depth-Limited Re-Solving)

```python
def resolve(game_state, blueprint, vnet, time_budget):
    tree = build_public_tree(game_state, depth_limit=STREET_END)
    R, S, sigma = warm_start_from_blueprint(tree, blueprint)

    deadline = now() + time_budget
    while now() < deadline:
        pi = forward_reach(tree, sigma)

        # GPU leaf eval (overlapped with CPU prep for next iter)
        leaves = get_leaves(tree)
        batch = build_features(leaves, pi, game_state)
        with cuda_stream(0):
            leaf_ev = vnet.infer(batch)

        cfv = backward_cfv(tree, leaf_ev, sigma, pi)
        R = maximum(R + instant_regret(cfv, sigma, pi), 0)
        sigma = regret_matching(R)
        S += sigma

        prune_low_reach(tree, pi, threshold=1e-5)

    return normalize(S[root_infoset])
```

---

## 13. Key Papers to Read

- **Libratus** (Brown & Sandholm 2017): blueprint + real-time re-solving architecture for heads-up
- **Pluribus** (Brown & Sandholm 2019): multiway 6-max; nested subgame solving; blueprint via MCCFR
- **DCFR** (Brown & Sandholm 2019): discounted CFR, faster convergence for blueprint training
- **DREAM** (Steinberger et al. 2020): deep regret minimization, GPU-native, value network training
- **Potential-Aware Abstraction** (Gilpin et al. 2007): card bucketing that preserves draw value
- **ReBeL** (Brown et al. 2020): recursive belief-based learning, blueprint + re-solving unified