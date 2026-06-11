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

---

## 3. Abstraction (Required for 6-max NLHE)

Full 6-max NLHE is unsolvable without abstraction. You need both card and action abstraction.

### 3.1 Action Abstraction

Discretize betting into a small fixed set per node:
- Example: `{fold, check/call, bet 33%, bet 75%, bet 150%, all-in}`
- Vary by street (preflop tighter, postflop richer)
- **Vary by position**: a 3x open from UTG and a 3x open from BTN are strategically different — the range compositions differ, the multiway pot probability differs, and the postflop position is different. Using the same abstract action tree for all positions is a meaningful abstraction quality loss, especially preflop where positional ranges diverge significantly. Recommended minimum: separate preflop trees for early position (UTG/UTG+1), middle position (MP/HJ), late position (CO/BTN), and blinds (SB/BB). Postflop: at minimum separate IP (in-position) and OOP (out-of-position) bet sizing sets.
- Constraint by SPR, pot size, min-raise rules

Action count directly controls solver cost: fewer actions = faster convergence. Quality matters more than quantity.

### 3.2 Card / Information Abstraction

**Preflop**: 169 canonical hands (suit isomorphism). No bucketing needed at this scale.

**Postflop suit isomorphism**: Suit isomorphism extends beyond preflop but is significantly more complex. On the flop, two boards are isomorphic only if there exists a suit permutation mapping one to the other that preserves all players' hole card suits simultaneously. For example, A♠K♠Q♥ and A♥K♥Q♠ are isomorphic (swap spades/hearts); A♠K♠Q♠ (monotone) is a distinct canonical class. Correct postflop isomorphism reduces the number of distinct flop textures from 22,100 to ~1,755 canonical boards. Implementing this requires:
- Canonical board normalization: sort suits by frequency, then by rank, assign canonical suit labels.
- Per-infoset isomorphism check at flop/turn/river construction time.
- Shared regret tables for isomorphic board classes.

This is non-trivial to implement correctly but cuts postflop infoset count by 10–12× on average. Most production solvers implement this; skipping it is a significant memory waste. The canonical normalization must be applied consistently across blueprint generation, value network training data, and runtime re-solving — any mismatch causes silent lookup errors where two isomorphic boards get separate regret entries and the blueprint diverges.

**Canonical suit normalization algorithm (concrete):**
1. Count the frequency of each suit across the board cards.
2. Sort suits by descending frequency. Break ties by the highest rank of any card in that suit (higher rank = earlier in canonical order). Break remaining ties arbitrarily but deterministically (e.g., alphabetical: c < d < h < s).
3. Assign canonical suit labels 0–3 in that sorted order.
4. Apply the same suit permutation to all hole cards in the infoset.
5. The result is the canonical form. Two boards/infosets are isomorphic if and only if they produce the same canonical form.

Example: board A♠K♠Q♥. Spades appear twice (highest freq), hearts once. Highest spade rank = A (rank 14). Canonical: spades → suit 0, hearts → suit 1. Canonical board: A0K0Q1. Board A♥K♥Q♠ maps to same canonical form. Board A♠K♥Q♦ has three different suits (freq 1 each); break ties by highest rank per suit: A♠ (rank 14) > K♥ (rank 13) > Q♦ (rank 12), so spades → 0, hearts → 1, diamonds → 2. Canonical: A0K1Q2.

This algorithm must be a single shared function imported by the blueprint builder, VNet data generator, and runtime resolver. Do not re-implement it in each module.

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

**Why:**

- Multiway Nash equilibrium doesn't decompose cleanly like heads-up
- Each player's strategy depends on the joint strategy of all others simultaneously
- Tree size explodes exponentially with players

**What solvers actually do:**

- Treat multiway as iterated 2-player subgames (wrong but tractable)
- Use aggressive abstraction (card buckets, limited bet sizes)
- Run CFR until "convergence" but the solution isn't a true Nash equilibrium

**In practice:**

- GTO Wizard, Solver+ etc. just run CFR on the abstracted tree and call it solved
- The result is "good enough" vs humans because humans are even further from optimal
- Multiway spots are just less exploited by bots, more human edge exists there

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

**Card removal (blocker filtering) — required at every chance node:**

When a community card is dealt, you must zero out any bucket whose hole cards conflict with the new board card before normalizing. Failing to do this causes ranges to contain impossible holdings, which makes range sums exceed 1 and produces incorrect EVs throughout the tree.

```python
def apply_card_removal(range_vec, bucket_holdings, new_board_cards):
    """
    range_vec: float32[NumBuckets]
    bucket_holdings: list of (card1, card2) representative hands per bucket
    new_board_cards: list of cards just dealt
    """
    board_set = set(new_board_cards)
    for b, (c1, c2) in enumerate(bucket_holdings):
        if c1 in board_set or c2 in board_set:
            range_vec[b] = 0.0
    total = range_vec.sum()
    if total > 1e-9:
        range_vec /= total
    # else: impossible state — all holdings blocked. Flag as error.
```

Apply this at flop deal, turn deal, and river deal for all active players' ranges. Also apply it to your own range when computing self-reach probabilities. For bucketed hands (multiple hole card combinations map to one bucket), zero the bucket only if ALL representative hands in that bucket conflict with the board — or more precisely, weight the bucket by the fraction of combinations that survive removal.

For postflop suit-isomorphic boards (§3.2), card removal must operate on the canonical hand representation, not the raw dealt cards.

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

**Decision rule:**

Before starting, estimate peak VRAM cost:
```
regret_bytes   = num_infosets * max_actions * 4
reach_bytes    = num_nodes * num_buckets * 4
strategy_bytes = num_infosets * max_actions * 4  # S table
adjacency_bytes = num_edges * 8

total_est = regret_bytes + 2 * reach_bytes + strategy_bytes + adjacency_bytes
```
If `total_est < 12GB`: use Strategy B (matrix-op CFR+ on GPU, all arrays resident).
If `total_est > 12GB` but `< 40GB`: tile reach/value arrays by subtree depth, keep regret/strategy tables in VRAM, stream reach passes in chunks.
If `total_est > 40GB`: switch to Strategy A (MCCFR + GPU batch eval). Regret tables stay in CPU RAM; only the VNet and current leaf batch go to GPU.

Recheck this estimate after any change to action abstraction or bucket count — both scale the tree linearly.

### 5.3 Strategy A — MCCFR + GPU Batch Evaluation

CPU orchestrates tree traversal (sampling). GPU accelerates:
- Leaf node evaluation (neural value net, large batches)

**Regret accumulation stays on CPU.** MCCFR regret updates are per-trajectory: each trajectory updates a specific set of infosets with values that depend on the exact sampled path. You cannot batch regret updates across trajectories the way you batch leaf evals — each trajectory touches different infosets with different values, so there is no regular structure to parallelize. Attempting to move regret accumulation to GPU gives no speedup and adds PCIe transfer overhead.

GPU work in MCCFR is strictly: leaf node EV inference via VNet, batched across many trajectories. Nothing else in the MCCFR inner loop belongs on the GPU.

Parallelism: run many independent MCCFR trajectories in parallel CPU threads. Collect leaf eval requests, dispatch to GPU in batches of 4096+. See §5.8 for thread safety on shared regret tables.

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

**SpMV kernel notes:** standard `torch.sparse.mm` or cuSPARSE `cusparseScsrmv` work for regular trees. Poker trees are irregular (fan-out varies by node type and action count), which causes warp divergence and poor memory access patterns in naive CSR SpMV. Practical mitigations:
- Sort nodes within each level by fan-out before building CSR (improves coalescing).
- Use ELL format instead of CSR for levels with near-uniform fan-out (common on lower streets where action abstraction is fixed).
- For the backward pass (CSR transpose), build and store the transpose CSR explicitly during preprocessing rather than computing it at runtime.
- cuSPARSE Generic API (`cusparseSpMM`) with `CUSPARSE_SPMM_ALG_DEFAULT` selects a tuned algorithm per shape; prefer it over direct kernel calls unless you have profiling data showing a specific kernel is better.
- Reference: the gpucfr repo (linked above) has a working CUDA SpMV for poker trees; study the level scheduling before writing your own.

```python
# Python + CUDA (PyTorch/CuPy style)
for level in levels_bottom_up:
    nodes = level_nodes[level]
    # backward: cfv[parent] += sigma[I,a] * cfv[child] * reach_opponent
    # adjacency[level] is a CSR matrix on GPU (torch.sparse_csr_tensor)
    cfv = sparse_backward_cuda(adjacency[level], cfv_children, sigma, reach)

# parallelized over all infosets via GPU kernel
instant_r = compute_instant_regret_cuda(cfv, reach, sigma, infoset_offsets)
R = torch.clamp(R + instant_r, min=0.0)   # CFR+ floor
S += t_weight * regret_matching_cuda(R, infoset_offsets)
```

**Memory constraints**: on 16GB VRAM (RTX 5080), this limits abstract game size. Budget:
- Regret table: `num_infosets * max_actions * 4 bytes`
- Reach/value arrays: `num_nodes * num_buckets * 4 bytes`
- If this exceeds VRAM, fall back to MCCFR + GPU eval (Strategy A)


### 5.6 Exploitability Estimation via Sampled Best Response

Exploitability = how much a best-response opponent can gain per game against your strategy. True exploitability requires computing an exact best response over the full tree, which is intractable for large abstractions. Use sampled BR instead.

**Algorithm (sampled BR):**
1. Fix your current average strategy `sigma`. **When using DCFR, this is the weighted average strategy `S_weighted = sum_t w(t) * sigma_t / sum_t w(t)`, not the simple cumulative average. Evaluating the wrong object (e.g., the current iterate `sigma_t` or the unweighted sum) gives misleading exploitability numbers — the current iterate is far from Nash at any individual step; the unweighted average converges slower than the weighted average under DCFR.**
2. For each player `i`, compute an approximate best response `BR_i` by traversing the tree and greedily selecting the action with the highest counterfactual value at each infoset, using MCCFR sampling to cover a fraction of the tree.
3. Estimate exploitability as: `eps = sum_i [ EV(BR_i, sigma_{-i}) - EV(sigma_i, sigma_{-i}) ] / 2`
4. Normalize by pot size for interpretability (mbb/hand or % of pot).

**Implementation notes:**
- Run sampled BR every K iterations (e.g., every 10,000 CFR iterations during offline training).
- Use a separate MCCFR traversal in best-response mode: at the BR player's nodes, take argmax instead of regret-matching; at opponent nodes, sample from current sigma.
- Sample ~1M–10M terminal nodes per estimate. More samples = lower variance estimate.
- Log convergence: exploitability should decrease monotonically with iterations under CFR+/DCFR. If it plateaus or increases, check for bugs in regret update or strategy accumulation.
- Exploitability in the abstracted game does not equal exploitability in the full game. A strategy with 0.1% pot exploitability in the abstraction can be significantly more exploitable outside it (abstraction bleeding). Track both if possible.

**Runtime check (simplified):**
- At runtime, after re-solving, run 100–500 sampled BR trajectories against the returned strategy.
- If estimated exploitability exceeds a threshold (e.g., 5% pot), fall back to blueprint strategy for that spot.
- This adds <50ms latency if BR sampling is vectorized.

### 5.7 Offline Training Pipeline

```
1. Build abstract game (action + card abstraction)
2. Initialize regret/strategy tables to zero (or uniform)
3. Run CFR+/DCFR/DREAM for N iterations:
   a. GPU: batch leaf eval / matrix passes
   b. CPU: sampling, scheduling, pruning (for MCCFR)
4. Every K iterations:
   a. Checkpoint (see below)
   b. Compute exploitability estimate (sampled)
   c. Log convergence
5. Export blueprint tables
6. Optional: distill blueprint into neural policy (behavior cloning + fine-tuning)
```

**Checkpointing and resume:**

Blueprint training runs for days or weeks. Without checkpointing, any interruption restarts from scratch. What to save and why:

```python
checkpoint = {
    'iteration': t,                  # REQUIRED: DCFR weights depend on t; wrong t = wrong weights
    'R': R.cpu(),                    # regret table (fp32)
    'S': S.cpu(),                    # strategy sum table (fp32)
    'sigma': sigma.cpu(),            # current strategy (can be recomputed from R, but save for speed)
    'rng_state': torch.get_rng_state(),          # CPU RNG
    'cuda_rng_state': torch.cuda.get_rng_state(), # GPU RNG (for MCCFR sampling reproducibility)
    'prune_mask': prune_mask.cpu(),  # if using regret-based pruning (§5.9)
}
torch.save(checkpoint, f'checkpoint_iter_{t}.pt')
```

**Why each field matters:**
- `iteration`: DCFR weight `w(t) = t^alpha / (t^alpha + 1)` depends on the true iteration count. Resuming at wrong `t` corrupts all future weighted averages in `S`. This is the most common resume bug.
- `rng_state` + `cuda_rng_state`: without these, MCCFR sampling is not reproducible after resume. Not critical for correctness (MCCFR converges regardless), but useful for debugging divergence.
- `prune_mask`: without this, pruned infosets get re-explored unnecessarily for the first few iterations after resume.

**Checkpoint frequency:** every 5,000–10,000 iterations for long runs. Keep the last 3 checkpoints (rolling window) to protect against corrupted saves. Checkpoint files for 10M infosets are ~80–280MB depending on dtype; disk space is not a concern.

### 5.8 Parallelism Across CPU and GPU (Offline)

- Use CUDA streams: stream 0 runs current batch, stream 1 prepares next
- CPU threads handle: sampling new trajectories, pruning, range normalization, scheduling
- Overlap CPU prep with GPU kernel execution at all times
- For MCCFR: multiple CPU threads each run independent traversals, coalesce leaf eval requests into GPU batches

**Thread safety for parallel MCCFR regret updates:**

When multiple CPU threads run independent MCCFR trajectories and write to shared regret tables, you have a data race. Do not use a global lock — it serializes all updates and eliminates parallelism. The correct approaches:

**Option A — Per-thread accumulators + periodic merge (recommended):**
```python
# Each thread maintains its own local regret delta array
local_R_delta = np.zeros(total_actions, dtype=np.float32)

# Thread accumulates deltas from its trajectories
for traj in thread_trajectories:
    local_R_delta += compute_regret_delta(traj)

# Periodic merge (e.g., every 1000 trajectories) with atomic add
with merge_lock:
    global_R += local_R_delta
    local_R_delta[:] = 0
```
This minimizes lock contention. Merge lock is held for a single bulk add, not per-trajectory.

**Option B — Lock striping:**
Partition infosets into B buckets (e.g., B = 256). Each bucket has its own lock. A thread locks only the bucket containing the infosets it is updating. Reduces contention by 1/B.

**Option C — Atomic float adds (acceptable for GPU, avoid on CPU):**
`std::atomic<float>` on x86 requires a compare-exchange loop (no native float atomic add before C++20). High contention makes this slower than Option A for shared infosets. Use only if infoset access patterns are nearly disjoint across threads.

**Recommended:** Option A with merge every 500–2000 trajectories depending on trajectory length. The slight staleness of the strategy (threads read global sigma that is a few merges behind) is acceptable for MCCFR — it is equivalent to running a few extra iterations before syncing.

### 5.9 Regret-Based Pruning (MCCFR, Required)

For MCCFR (Strategy A), regret-based pruning is not optional — it typically cuts 60–80% of tree traversals and is necessary to reach convergence in reasonable time.

**Algorithm (Brown & Sandholm 2015):**
At each player node during traversal, before recursing:
1. Compute the sum of positive regrets: `R_plus = sum_a max(0, R[I,a])`
2. For each action `a`: if `R[I,a] < -PRUNE_THRESHOLD` and `R_plus > 0`, skip this subtree entirely.
3. Do not prune if ALL regrets are negative (no positive baseline to compare against).

**Threshold choice:** `PRUNE_THRESHOLD` is typically a small negative value proportional to the pot size at that node — e.g., `-0.01 * pot`. Too aggressive (large threshold) prunes actions that haven't converged yet. Too conservative (near zero) provides no speedup. Start at `-0.01 * pot` and tighten if convergence is good.

**When to enable:** do not prune for the first `T_min` iterations (e.g., first 1000 iterations). Regrets need time to stabilize before pruning is safe.

**Implementation:** track a `prunable[I,a]` boolean per infoset-action. Set it when `R[I,a]` has been below threshold for `K` consecutive checkpoints (e.g., K=3). Clear it whenever `R[I,a]` rises above zero (action becomes relevant again).

**GPU note:** for matrix-op CFR (Strategy B), pruning is applied differently — zero out rows of the adjacency matrix for pruned subtrees rather than skipping traversal. Rebuild pruned CSR every N iterations.

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

**Range initialization overview:** opponent range handling has two distinct steps that must happen in order:
1. **§6.4 (this section):** one-time initialization at the start of each re-solve — load blueprint regrets/strategy and set the opponent's current range based on all actions observed so far in the hand.
2. **§6.9:** ongoing per-action update throughout the hand — after each observed action, apply a Bayesian update to the range before the next re-solve.

These are not interchangeable. §6.4 sets the starting range for a given re-solve. §6.9 keeps that range current between re-solves.

Before beginning re-solving:
1. Load blueprint regrets/strategy for all infosets in the current subtree into CPU memory (or VRAM if budget allows)
2. Initialize `R[I,a]` from blueprint regrets (CFR+ warm-start)
3. Initialize `S[I,a]` from blueprint average strategy
4. **Initialize opponent ranges from observed action history** (see below)
5. Run re-solving iterations until time budget

This typically converges 10–50× faster than cold-start.

**Opponent range initialization (required):**

Do not initialize opponent ranges to uniform. By the time you reach a re-solving decision point, you have observed a sequence of actions. Each observed action is a Bayesian update on your opponent's range:

```python
# After observing opponent play action a at infoset I:
for b in range(num_buckets):
    opponent_range[b] *= blueprint_sigma[I, a, b]  # P(bucket b | action a)
opponent_range /= opponent_range.sum()              # normalize
```

Apply this update for every action in the hand history before warm-starting. At a flop decision point after preflop action UTG-raise / BB-call, apply two Bayesian updates: one for the raise, one for the call.

If a bucket's range mass drops below `1e-6`, zero it out and renormalize (numerical stability). If the opponent played an action with near-zero blueprint probability for some bucket, that bucket is essentially removed from their range.

Skipping this step means re-solving always assumes a uniform starting range, which discards all information from the hand history and significantly degrades decision quality.

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

**Warning: unsafe re-solving and manipulation.**

Depth-limited re-solving without safe subgame solving (as described in Libratus §4, "safe endgame solving") is exploitable in theory: an opponent who knows you re-solve can choose lines that lead to subgames where your re-solved strategy is weaker than the blueprint. This is the "unsafe re-solving" problem.

In practice the risk is low for most use cases (human opponents do not know your solver architecture and cannot precisely exploit it), but be aware of it:
- Unsafe re-solving means your strategy in the full game is not guaranteed to be an improvement over the blueprint, even if the subgame re-solve looks good in isolation.
- Safe re-solving requires computing counterfactual values for the "gift" given to the opponent (the trunk strategy that led to this subgame), which is significantly more expensive.
- For a production bot, safe re-solving is the correct approach. See Brown & Sandholm (2017) Libratus supplementary for the algorithm.
- For a research/study tool, unsafe re-solving is acceptable.

### 6.6 Action Translation

When the live game presents a bet size not in your action abstraction, you must map it to a nearby abstract action before looking up the blueprint or running re-solving. This is called action translation, and getting it wrong produces exploitable behavior.

**Problem:** if a player bets 47% pot and your abstraction only has {33%, 75%}, naively rounding to 33% systematically underestimates the threat; rounding to 75% overestimates it.

**Standard approach (pseudo-harmonic mapping):**
- Map the real bet `b` to a probability-weighted mixture of the two bracketing abstract bets `b_lo` and `b_hi`.
- Pseudo-harmonic weights (Brown & Sandholm, Libratus): `w_hi = (b - b_lo) / (b_hi - b_lo)` adjusted by harmonic interpolation to avoid linear bias at extremes.
- Apply the blended strategy: `sigma_translated = (1 - w_hi) * sigma[b_lo] + w_hi * sigma[b_hi]`

**All-in translation:** if the real bet exceeds the largest abstract bet, use the all-in action. Do not extrapolate.

**Consistency requirement:** the same translation function must be used during:
- Blueprint lookup (before warm-start)
- Re-solving initialization
- Opponent range update after observing their action

If any of these use different translation logic, opponent ranges will drift from reality.

**Reference:** Libratus supplementary material covers pseudo-harmonic mapping in detail. Implement and test this before using the blueprint at a real table.

### 6.7 GPU Overlap in Runtime

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

### 6.9 Opponent Range Update Protocol

At every observed action during the hand, update opponent ranges before the next re-solve. This is distinct from the warm-start initialization (§6.4) — it must happen continuously throughout the hand, not just once.

**Per-action Bayesian update:**
```python
def update_range_on_action(range_vec, blueprint_sigma, infoset_id, action_idx):
    # range_vec: float32[NumBuckets], opponent's current range
    # blueprint_sigma: float32[NumBuckets] per action at this infoset
    #   (or re-solved sigma if available for this node)
    likelihoods = blueprint_sigma[infoset_id, action_idx]  # P(action | bucket)
    range_vec *= likelihoods
    total = range_vec.sum()
    if total < 1e-9:
        # opponent played an off-tree action; range has collapsed
        # fall back to uniform or raise a flag for action translation
        range_vec[:] = 1.0 / num_buckets
    else:
        range_vec /= total
```

**What strategy to use for likelihood:**
- If the action was in the abstraction and the current node was re-solved: use the re-solved sigma.
- If the action was in the abstraction but not re-solved: use the blueprint sigma.
- If the action was off-abstraction: apply action translation (§6.6) to get a blended sigma, then update.

**Storage:** maintain one `float32[NumBuckets]` range vector per opponent per street. Reset to uniform at the start of each hand. Save range state after each action so re-solving can be warm-started from the correct range at any point in the hand.

**Range coherence across streets:** when a new community card is dealt, ranges do not reset — they carry forward. The card is a chance event that filters buckets: remove any bucket whose hole cards conflict with the new board card (impossible holdings). Renormalize after removal.

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

**The value network is the final layer of the system and is optional.**

Without it, you must either solve to terminal nodes (only feasible for very small trees) or use Monte Carlo rollouts from leaf nodes (high variance, not real-time compatible). For real-time re-solving on a postflop subgame, a VNet is a practical necessity. For offline blueprint generation with full tree traversal (no depth limit), you do not need one.

Build order recommendation:
1. Build and validate the offline CFR solver without a VNet (full tree depth, no leaf approximation).
2. Generate ground-truth training data from step 1.
3. Train the VNet on that data.
4. Integrate the VNet as a leaf evaluator in the runtime re-solving pipeline.
5. Regenerate data with the VNet-assisted solver and iterate.

The system is functional and produces a valid blueprint at step 1. Steps 2–5 are required only for real-time re-solving (the runtime layer).

### 7.2 Architecture

Input features per leaf node:
- Board cards (one-hot or suit-normalized encoding, using canonical suit normalization from §3.2)
- Pot size, stack sizes (normalized)
- Street
- **Position:** encode each active player's position relative to the dealer button (e.g., one-hot over {BTN, SB, BB, UTG, MP, CO} or ordinal 0–5). Position is critical for EV — the same board/stack/pot situation has a different EV profile depending on who acts first postflop. Omitting position causes systematic errors in out-of-position vs. in-position spots, which are among the most common and strategically significant situations in 6-max.
- **Action history encoding (fixed-width sequence, player-tagged):** encode each action as a tuple `(player_seat, action_type, bet_size_normalized)`. Player seat must be included — a 3-way pot where UTG bets and BTN calls has a different EV profile than BTN bets and UTG calls, but an encoding without player identity cannot distinguish them. Use a fixed-length sequence padded to max history length (e.g., 20 actions). Represent each action as a small embedding or one-hot over `(seat, action_type, size_bucket)`.
- Player 0 range: `float32[NumBuckets]`
- Player 1 range: `float32[NumBuckets]`
- (Multiway: one range per player, pad to max_players with zeros for folded seats; include a `active_players` bitmask)

Output:
- `EV[player]` for each active player. **Constraint: `sum(EV[player]) == current_pot` at the depth-limited leaf node, not pot at game end.** The pot at a depth-limited leaf has not been distributed yet; EVs represent each player's expected share of the chips currently in the middle plus their remaining stack contributions. This is distinct from terminal-node EV which sums to the total starting stacks.
- Or: advantage value `A[player, bucket]` (richer, more expensive)

**Enforcing the sum constraint during training:**
```python
# After VNet forward pass, project outputs onto the constraint hyperplane:
ev_raw = vnet(features)                        # shape: [batch, num_players]
ev_sum = ev_raw.sum(dim=1, keepdim=True)
ev_corrected = ev_raw - (ev_sum - current_pot.unsqueeze(1)) / num_players
# ev_corrected.sum(dim=1) == current_pot for all samples
```
Add this projection as a post-processing step at both training time (apply before computing loss) and inference time. Also add a sanity assert during training: if `abs(ev_raw.sum(dim=1) - current_pot).max() > 0.1 * pot`, log a warning — large violations indicate a feature engineering bug (wrong pot normalization or missing stack inputs).

Architecture: MLP with 4–8 layers, residual connections, LayerNorm. Wider is better given GPU budget.

### 7.3 Training Data Generation

**Option A (Solver rollouts) — recommended:**
- Run offline CFR to convergence on many board/range samples
- Record (state, both ranges) → (EV per player) pairs
- ~10M–100M samples needed for good coverage
- Ground truth quality; the only reliable approach

**Bootstrapping problem:** the first training run has no VNet, so you cannot use depth-limited re-solving during data generation. For the initial dataset, run CFR to full tree depth (no depth limit, no VNet) on a set of sampled subgames. This is slow but produces clean ground truth. Once you have a trained VNet v0, regenerate data using depth-limited solving with v0 at leaves, producing a better dataset for v1. Iterate 2–3 rounds. Do not skip this: a VNet trained only on blueprint rollouts (Option B) learns to approximate a weak baseline, and the error compounds through re-solving.

**Option B (Bootstrap from blueprint) — acceptable as warm-start only:**
- Use blueprint strategy to simulate rollouts forward
- Cheap but biased toward blueprint quality
- Do not use as sole training signal; bias compounds and the value network learns to approximate a weak baseline, not true EV

**Option C (Self-play / DREAM-style) — not recommended for this architecture:**
- DREAM generates (state, regret) samples during training and trains a regret network simultaneously
- Works in theory but requires careful implementation: if the regret network is wrong, CFR updates are wrong, which corrupts future training data
- In practice, self-play bootstrapping in poker tends to collapse to exploitable strategies or plateau far from Nash without solver-generated ground truth as a corrective signal
- Avoid for production; use solver rollouts (Option A) as the primary signal

**A note on value networks as leaf evaluators:**
A value network at the depth limit is the tractable alternative to full tree rollout. The VNet approximates the EV of continuing play, which means errors in its training data propagate into re-solving quality. Alternatives at the depth limit:
- **Longer depth limit with no VNet**: more accurate but runtime explodes
- **Nested subgame solving at leaves**: correct but recursively expensive
- **Monte Carlo rollouts from leaves**: unbiased but high variance; not real-time compatible

For 2–8 second decision budgets, a TensorRT-compiled MLP value network is the only practical option on current hardware. Accept the approximation error; minimize it via iterative bootstrapped training (see bootstrapping note above).

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

**Cold-start latency:** the VNet latency figure of `< 5ms per batch at 4096 leaves` (§7.4) assumes the model is already warmed in VRAM. First inference after process start on a cold TensorRT engine can be 50–200ms due to kernel autotuning and CUDA context initialization. Always call a dummy forward pass (e.g., `vnet.infer(zeros_batch)`) at game start before any real decision. Do this during the waiting period before cards are dealt, not at the first decision point.

---

## 9. Non-Negotiables for GPU Performance

- Flat arrays, not pointer graphs or hash maps
- Precomputed offsets for infoset/action storage
- Large batches (≥ 1024, ideally 4096+) for GPU inference
- Minimal branch divergence in GPU kernels
- No global atomics on shared regret tables in inner loops (use per-block accumulators + reduction)
- Overlap CPU prep with GPU execution via CUDA streams
- Float32 for regrets and strategy sums; bfloat16 for reach/value arrays; fp16 for VNet inference (see §11.1 for full precision tradeoff analysis). Note: reach vectors (`pi`) are safe in either fp16 or bfloat16 since values stay in [0,1]; prefer bfloat16 for cfv due to its wider exponent range. Do not use fp16 for raw cumulative regrets.
- Keep action count small (5–9 per node); action abstraction quality over quantity
- Blueprint warm-start in runtime mode (non-negotiable for convergence speed)

---

# 10. Multiway Solving (3+ Active Players)

True multiway equilibrium (3+ players) is not computationally feasible for 6-max NLHE. The following section describes the practical approximations used in production systems, ordered from simplest to most sophisticated.

---

### 10.1 Why Multiway Is Hard

*   **Non-uniqueness:** Nash equilibrium is not unique in 3+ player games. CFR convergence does not guarantee a single useful strategy.
*   **Joint action space:** Counterfactual values must be computed over the joint distribution of all opponents' actions. The action space grows as O(|A|^n) where n is the number of active players.
*   **Range interdependence:** Each player's strategy affects all others simultaneously. Ranges cannot be decomposed into independent 2-player matchups.
*   **Coalition effects:** Multiple opponents can implicitly coordinate (e.g., two players applying pressure to force a fold) even without explicit collusion.

---

### 10.2 Approximation Hierarchy

| Method | Description | Complexity | Use Case |
|--------|-------------|------------|----------|
| **A. Sequential 2-Player Subgames** | Pick a focal player; treat all others as a single aggregated opponent. Solve 2-player game; rotate focal player. | Low | Baseline, early implementation |
| **B. Coalition Approximation** | Group all opponents into a single "team" with combined range and shared objective. Solve hero vs. coalition. | Medium | Production runtime (Pluribus-style) |
| **C. Full Multiway CFR** | Run CFR on the full multiway tree with joint action regret updates. | Very High | Small abstractions only, research |
| **D. Learned Multiway Value Function** | Neural network predicts EV for each player given all ranges. Implicitly learns multiway interaction. | High (training) | Modern approach, scales to any player count |

---

### 10.3 Method A: Sequential 2-Player Subgames

**Algorithm:**
1. At a multiway node with N active players, select a focal player i.
2. Aggregate all other players' ranges into a single opponent range: R_opponent = normalize(sum_{j != i} R_j).
3. Solve the 2-player subgame: focal player i vs. R_opponent.
4. Record the resulting strategy for player i.
5. Rotate focal player and repeat until all players have a strategy.

**Pros:**
*   Reuses existing 2-player solver without modification.
*   Simple to implement and debug.

**Cons:**
*   Ignores coordination between opponents.
*   Can produce incoherent strategies (e.g., two players both overfolding because each assumes the other will apply pressure).
*   No equilibrium properties; strategies may be mutually inconsistent.

**When to use:**
*   Prototype phase.
*   Nodes with 4+ players where any approximation is acceptable.

---

### 10.4 Method B: Coalition Approximation (Recommended for Runtime)

**Algorithm:**
1. Group all opponents into a single coalition agent.
2. The coalition's range is the union (or weighted sum) of individual opponent ranges.
3. The coalition's objective is to maximize the total EV of its members, or equivalently, minimize the hero's EV.
4. Solve the 2-player game: hero vs. coalition.
5. Map the coalition's strategy back to individual players proportionally to their range weights.

**Implementation details:**
*   **Range aggregation:** `R_coalition[b] = sum_{j != hero} R_j[b]` for each bucket b. Normalize so that `sum(R_coalition) = 1`.

    **Important caveat — double-counting:** summing ranges across opponents treats the coalition as a single player who can hold bucket b. In reality, two opponents can simultaneously hold different hole cards that both map to bucket b. The summed range over-represents the coalition's threat for buckets that appear in multiple opponents' ranges. This is an approximation, not a correct probability. The correct treatment would compute joint probabilities over all opponent hand combinations, which is intractable. Accept this approximation and its consequence: coalition EV will be slightly overestimated for high-frequency buckets. Validate using range coherence checks (§10.8).

*   **Action mapping:** If the coalition strategy specifies action a with probability p, each opponent j plays action a with probability `p * (R_j[b] / R_coalition[b])` at bucket b. This back-mapping is also approximate — it assumes opponents act proportionally to their contribution to the coalition range, not independently.
*   **Payoff:** Hero's EV is computed against the coalition's joint strategy. Coalition EV is the sum of individual EVs.

**Pros:**
*   Better than sequential subgames; captures some multiway pressure.
*   Compatible with blueprint warm-start (coalition strategy initialized from blueprint aggregate).
*   Fast enough for real-time re-solving.

**Cons:**
*   Coalition can over-represent threat (e.g., 3 opponents each with 30% equity do not combine to 90% threat).
*   Individual opponent strategies may not be best responses to each other.
*   Not a true equilibrium; exploitable in theory.

**When to use:**
*   Default for 3-way and 4-way nodes in runtime re-solving.
*   When combined with a learned value function for leaf evaluation.

---

### 10.5 Method C: Full Multiway CFR

**Algorithm:**
1. Build the full public tree for all active players.
2. At each player node, compute counterfactual values over the joint distribution of all opponents' strategies.
3. Update regrets using the joint counterfactual values.
4. Apply regret matching to derive new strategies.

**Regret update formula (3-player example):**
*   For player 1 at infoset I with actions a:
    *   v_1(I, a) = sum_{a2, a3} [ sigma_2(I2, a2) * sigma_3(I3, a3) * u_1(z) * pi_{-1}(h) ]
    *   where pi_{-1}(h) is the product of all opponents' reach probabilities to reach terminal node z.
*   Instant regret: r_1(I, a) = v_1(I, a) - sum_b [ sigma_1(I, b) * v_1(I, b) ]

**Pros:**
*   Theoretically sound approximation of Nash equilibrium.
*   Strategies are mutually consistent.

**Cons:**
*   Joint action space is intractable for 6-max NLHE (even with abstraction).
*   Convergence is orders of magnitude slower than 2-player CFR.
*   Memory requirements scale with the product of opponent infosets.

**When to use:**
*   Research on toy games (Kuhn poker 3-player, small Leduc).
*   Not recommended for 6-max NLHE production.

---

### 10.6 Method D: Learned Multiway Value Function (Modern)

**Architecture:**
*   **Input:** Public state (board, pot, stacks, street, action history) + range vectors for all N active players.
*   **Output:** EV for each active player, or advantage A[player, bucket].
*   **Network:** MLP with 4-8 layers, residual connections, LayerNorm. Input dimension scales linearly with player count.

**Training data generation:**
*   **Option 1 (Solver rollouts):** Run multiway CFR (Method C) on small abstractions to convergence. Record (state, all ranges) -> (EV per player) pairs. Expensive but accurate.
*   **Option 2 (Self-play + DREAM):** DREAM generates (state, regret) samples during training. Train a value head simultaneously on multiway states.
*   **Option 3 (Bootstrap from coalition):** Use coalition approximation (Method B) to generate rollouts. Cheap but biased toward coalition quality.

**Pros:**
*   Scales to any number of players (2-6) with the same network.
*   Fast at runtime: single forward pass per leaf node.
*   Implicitly learns complex multiway interactions (squeeze play, implied odds from multiple callers, etc.).

**Cons:**
*   Requires massive training data (10M-100M+ multiway states).
*   Must generalize across player counts, stack depths, and board textures.
*   Quality depends on data distribution; rare multiway spots may be poorly estimated.

**When to use:**
*   Primary method for leaf evaluation in runtime re-solving.
*   Combined with coalition approximation (Method B) for the CFR layer.

---

### 10.7 Runtime Decision Logic

```
game state arrives with N active players
    |
    v
if N == 2:
    |   exact 2-player re-solve (standard CFR/CFR+/DCFR/MCFR)
    v
elif N == 3:
    |   coalition approximation (Method B)
    |   value network evaluates leaves (Method D)
    v
elif N >= 4:
    |   heuristic pre-filter (fold if range is weak, call if strong)
    |   coalition approximation for remaining players
    |   value network evaluates leaves
    v
return action distribution at root infoset
```

---

### 10.8 Validation Checks

Any multiway approximation must be checked for pathologies:

*   **Range coherence:** After solving, verify that no two players both hold the same strong hand with high probability. Sum of bucket probabilities across players should not exceed 1 for any single hand.
*   **Monotonicity:** Adding an opponent should never increase hero's EV. If EV increases, the approximation is inconsistent.
*   **Exploitability sampling:** Play the strategy against a best-response opponent. If exploitability exceeds 0.5 * pot size, the approximation has failed.
*   **Strategy consistency:** In coalition approximation, verify that mapped individual strategies are not dominated (e.g., folding a hand that is clearly the nuts).

---


### 10.9 Key Insight

Pluribus and other winning 6-max agents do not solve true multiway equilibrium. They solve 2-player and 3-player subgames with approximations, and rely on a strong value network to paper over the gaps. The goal is not optimality; it is to be unexploitable enough that opponents cannot systematically deviate for profit. Coalition approximation + learned value function achieves this at production scale.


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

### 11.1 Regret Table Compression: fp16 vs fp32

**The problem:** regret tables are the largest persistent structure in VRAM during training. At scale (10M+ infosets × 7 actions), fp32 costs 280MB+. Doubling action abstraction doubles this linearly.

**fp16 during training — use with caution:**
- fp16 has ~3 decimal digits of precision and clips at ±65504.
- Regrets accumulate over millions of iterations. Large positive regrets (common for dominant actions) will overflow fp16 near iteration 100k–1M depending on normalization.
- Small regrets (rare actions) underflow to zero, which permanently zeros their strategy weight. This causes strategy collapse for low-frequency actions.
- **Conclusion:** do not store raw cumulative regrets in fp16.

**Safe fp16 usage:**
- Reach vectors (`pi`): safe to use fp16 or bfloat16. Values stay in [0, 1]; precision loss has minimal CFR impact.
- Counterfactual values (`cfv`): safe to use bfloat16 (wider exponent range than fp16). Verify no overflow on your specific game scale.
- VNet inference: always fp16 or bfloat16. TensorRT fp16 is standard; no precision issue for a forward pass.
- Strategy sums (`S`): borderline. Use fp32 unless VRAM is critically constrained; the average strategy is your final output and precision matters.

**Recommended hybrid:**
```python
R = torch.zeros(total_actions, dtype=torch.float32, device='cuda')  # regrets: fp32
S = torch.zeros(total_actions, dtype=torch.float32, device='cuda')  # strategy sum: fp32
pi  = torch.ones(num_nodes * num_buckets, dtype=torch.bfloat16, device='cuda')
cfv = torch.zeros(num_nodes * num_buckets, dtype=torch.bfloat16, device='cuda')
# VNet input/output: fp16 (TensorRT handles internally)
```

**Blueprint export (post-training):**
- Quantize average strategy to uint16 per action (normalized per infoset, sum = 65535).
- Optionally quantize to uint8 for aggressive compression (coarser strategy, acceptable for low-reach infosets).
- Regrets for warm-starting: export as fp16 after clipping to [-1000, 1000] (values outside this range have negligible RM+ impact after normalization).
- Expected blueprint size: 1M infosets × 7 actions × uint16 ≈ 14MB. 10M infosets ≈ 140MB. Fits in VRAM easily for warm-starting.

**Quantization error tradeoffs:**

uint16 (2 bytes/action): ~4 decimal digits of precision per action probability. For a 7-action infoset, the worst-case rounding error per action is `1/65535 ≈ 0.0015%`. This is negligible for all infosets; use uint16 by default.

uint8 (1 byte/action): ~2 decimal digits of precision. Worst-case rounding `1/255 ≈ 0.4%` per action. Acceptable for infosets with low reach probability (rare board/range combinations that are almost never encountered). Not acceptable for high-reach infosets (preflop, common flop textures) — mixed strategies that should be close to 50/50 can round to 48/52 or 45/55, which is meaningful exploitability at high frequency.

**Recommended tiered compression:**
```
if infoset_reach > REACH_THRESHOLD:   # e.g. 1e-3
    store as uint16
else:
    store as uint8  # halves storage for rare spots
```
Track which infosets are uint8 vs uint16 in a separate bitmap. At warm-start, dequantize back to float32 before loading into the regret table.

**Quantization convergence check:** after quantizing and reloading the blueprint, run 1000 sampled BR trajectories and compare exploitability vs. the unquantized strategy. If delta > 0.1% pot on high-reach infosets, the quantization is too coarse for those spots.

**Precision convergence check:** if switching from fp32 to bf16 for reach/cfv, run a calibration: compare exploitability at iteration 100k with full fp32 vs hybrid. Difference should be < 0.05% pot. If larger, your game scale has precision sensitivity and you should keep cfv in fp32.

---

## 12. Preflop Solvability: HU, 3-Way, 6-Max

This section describes the computational boundaries of solving preflop scenarios in No-Limit Hold'em, ordered by player count. No method described here relies on learned regret models or neural regret approximation.

---

### 12.1 Heads-Up Preflop (2 Players)

**Status:** Solvable with standard abstraction.

**Hand space:** 1,326 distinct starting hands. Reduced to 169 canonical hands via suit isomorphism (e.g., all AKs combinations are equivalent preflop; suits only matter for flush/blocker potential, which is symmetric preflop).

**Tree size:** Approximately 10^6 to 10^7 information sets with the abstraction above. Fits comfortably in GPU memory (under 4GB for regret tables).

**Method:** CFR+ or DCFR on the abstracted tree. Converges in hours on a modern GPU (RTX 4080/5080 class).

**Output:** Near-Nash equilibrium strategy for the abstracted game. Exploitability typically under 0.5% of the pot against a best response in the same abstraction.

**Commercial availability:** Yes. PioSOLVER, GTO+, and similar tools solve HU preflop exactly within their abstractions.

---

### 12.2 Three-Way Preflop (3 Players)

**Status:** Marginally solvable with heavy abstraction.

**Hand space:** 1,326 hands per player. Joint hand combinations: 1,326^3 ≈ 2.3 × 10^9. Even with 169 canonical buckets per player, joint bucket combinations are 169^3 ≈ 4.8 × 10^6.

**Action space problem:** Each player acts in sequence with full knowledge of previous actions. A single orbit with 5 actions each creates 5^3 = 125 branches. Multiple orbits (limp, raise, call, 3-bet, call, 4-bet) multiply this further.

**Tree size:** 10^9 to 10^10 information sets with moderate abstraction. Exceeds single-GPU memory. Requires aggressive abstraction to fit.

**Required abstractions:**
*   **Card:** 169 canonical hands, possibly further reduced to 50-100 buckets by grouping similar hands (e.g., all broadways, all suited connectors).
*   **Action:** Severely restricted set. Example: {fold, call, raise 3bb, all-in}. No intermediate sizing.

**Method:** MCCFR with external sampling. CPU traverses sampled paths; GPU evaluates leaf nodes in batches. Alternatively, explicit CFR+ on a heavily pruned tree.

**Convergence:** Days to weeks on high-end hardware. Exploitability is higher than HU (2-5% of pot) due to abstraction coarseness.

**Commercial availability:** Limited. Some research solvers exist; no mainstream consumer tool solves 3-way preflop to high precision.

---

### 12.3 Six-Max Preflop (6 Players)

**Status:** Not solvable without abstraction. Full game is intractable.

**Hand space:** 1,326 hands per player × 6 players. Joint hand combinations: 1,326^6 ≈ 5.2 × 10^18. Even with 169 canonical buckets: 169^6 ≈ 2.4 × 10^13 joint bucket states.

**Action space problem:** 6 players act in sequence. Early position faces 5 unknown opponents. Each orbit multiplies branches by 5^6 = 15,625 for a single action per player. Multiple betting rounds (limp, raise, 3-bet, 4-bet, 5-bet, all-in) compound this.

**Information set count:** Estimated 10^15 to 10^18 for the full game. Even with aggressive abstraction, 10^12 to 10^14.

**Memory requirement for explicit regret tables:**
*   10^12 infosets × 5 actions × 4 bytes = 20 TB.
*   Far beyond any existing GPU or CPU memory system.

**What is actually done:**
*   **Blueprint construction:** Solve an abstracted game offline using MCCFR or DCFR. The abstraction includes:
    *   169 canonical preflop hands (no further reduction, but suit isomorphism applied).
    *   Severe action abstraction: {fold, call, raise 3bb, all-in} or similar.
    *   Early position solved with reduced opponent models (e.g., assume late position plays a fixed strategy).
*   **Nested subgame solving:** At runtime, when the hand reaches a specific state (e.g., UTG raises, MP calls, hero in BB), build a smaller subgame with only the active players and solve it in real-time.
*   **Heuristic preflop charts:** Many systems use pre-computed charts (e.g., GTO charts for open-raising, 3-betting, calling) derived from smaller solved games, not from a full 6-max solve.

**Pluribus approach:**
*   Blueprint trained with MCCFR on an abstracted 6-max game.
*   Action abstraction: small discrete set per node.
*   Card abstraction: 169 canonical hands preflop; no postflop bucketing in the blueprint (postflop handled by real-time solving).
*   The blueprint is not a full equilibrium. It is a coarse approximation used to warm-start real-time solving.

**Commercial availability:** No consumer tool claims to solve full 6-max preflop. All "GTO" 6-max charts are abstractions, approximations, or interpolations from smaller solved games.

---

### 12.4 Summary Table

| Scenario | Players | Joint Hand Combos | Solvable? | Method | Time | Exploitability |
|----------|---------|-------------------|-----------|--------|------|----------------|
| HU preflop | 2 | 1.7 × 10^6 | Yes | CFR+ / DCFR | Hours | < 0.5% pot |
| 3-way preflop | 3 | 2.3 × 10^9 | Barely | MCCFR + heavy abstraction | Days-weeks | 2-5% pot |
| 6-max preflop | 6 | 5.2 × 10^18 | No — approximated only | Blueprint (MCCFR) + nested subgames | Offline weeks for blueprint | Unknown, high |


We don't solve 6max preflop. We approximate it.
6-max preflop has no exact solution. The blueprint is a coarse, sampled approximation. It is good enough to avoid major leaks and warm-start postflop play, but it is not Nash equilibrium.

**Build a blueprint offline:**
- 169 canonical hands per player (suit isomorphism).
- Severe action abstraction: {fold, call, raise 3bb, all-in}.
- MCCFR with external sampling. Traverse a tiny fraction of the tree, accumulate regrets.
- Accept that 99.9% of nodes are never visited.

**Store only what fits:**
- Regret table: compress, quantize, prune low-reach nodes.
- Blueprint is ~1-10GB, not 20TB. It covers common spots; rare spots use fallback heuristics.

**Use it at runtime:**
- Load blueprint for the current preflop spot (position, action history).
- If the spot is in the blueprint, play the stored strategy.
- If not, fall back to real-time coalition approximation or heuristic charts.

**Real-time re-solving is not used preflop**
- Too many nodes, ranges too wide.
- Re-solving is for postflop subgames where the tree is smaller.

---

### 12.5 Key Insight

The jump from 2 to 3 players is manageable with abstraction. The jump from 3 to 6 players is a wall. No existing system solves 6-max preflop exactly. All production systems use:

1.  A coarse offline blueprint (abstracted game, approximate equilibrium).
2.  Real-time subgame re-solving at decision points (smaller, tractable games).
3.  Heuristic charts for common spots (derived from smaller solves or expert tuning).

The blueprint does not need to be perfect. It needs to be good enough to warm-start real-time solving and avoid catastrophic errors.

---

## 13. Full Pseudocode

### Offline (Matrix-Op CFR+)

```python
# Python + CUDA (PyTorch). All tensors on GPU unless noted.
# Preprocessing (CPU → GPU once)
levels = topological_levels(tree)
adjacency = build_csr_cuda(tree)          # torch.sparse_csr_tensor per level
infoset_offsets = precompute_offsets(infosets).cuda()

R = torch.zeros(total_actions, device='cuda')   # regrets
S = torch.zeros(total_actions, device='cuda')   # strategy sums
sigma = uniform_strategy_cuda(infoset_offsets)

stream_fwd  = torch.cuda.Stream()
stream_bwd  = torch.cuda.Stream()

for t in range(1, num_iterations + 1):
    weight = dcfr_weight(t)

    # Forward: reach probabilities (GPU, level by level)
    pi = torch.ones(num_nodes * num_buckets, device='cuda')
    with torch.cuda.stream(stream_fwd):
        for level in levels_top_down:
            pi = sparse_forward_cuda(adjacency[level], pi, sigma)

    # Leaf evaluation (GPU, fp16, overlap with next forward setup)
    leaf_features = build_leaf_features_cuda(leaves, pi)
    with torch.cuda.stream(stream_bwd):
        leaf_ev = vnet_infer_fp16(leaf_features)   # TensorRT or torch.compile

    torch.cuda.synchronize()

    # Backward: counterfactual values (GPU)
    cfv = torch.zeros(num_nodes * num_buckets, device='cuda')
    cfv[leaves] = leaf_ev.float()
    for level in levels_bottom_up:
        cfv = sparse_backward_cuda(adjacency[level], cfv, sigma, pi)

    # Regret update (GPU kernel, parallelized over infosets)
    instant_r = compute_instant_regret_cuda(cfv, sigma, pi, infoset_offsets)
    R = torch.clamp(R + instant_r, min=0.0)        # CFR+ floor
    sigma = regret_matching_cuda(R, infoset_offsets)
    S.add_(weight * sigma)                          # DCFR weighted average

blueprint = normalize_strategy_cuda(S, infoset_offsets)
```

### Runtime (Depth-Limited Re-Solving)

```python
# Python + CUDA. CPU orchestrates; GPU handles VNet and large regret updates.
def resolve(game_state, blueprint, vnet, time_budget_sec):
    tree = build_public_tree(game_state, depth_limit=STREET_END)   # CPU
    R, S, sigma = warm_start_from_blueprint(tree, blueprint)        # CPU → GPU
    # warm_start_from_blueprint must do ALL of the following (in order):
    #   1. For each infoset in tree: look up blueprint by canonical infoset key
    #      (board must be suit-normalized per §3.2 before lookup)
    #   2. Apply action translation (§6.6) for any infoset whose actions differ
    #      from the blueprint abstraction (off-tree bet sizes in current game state)
    #   3. Dequantize blueprint_sigma from uint16/uint8 → float32 (§11.1)
    #   4. Initialize opponent range from hand action history via Bayesian updates
    #      (§6.4, §6.9) — do NOT use uniform range
    #   5. Return R (fp32), S (fp32), sigma (fp32) tensors sized for this subgame
    # Any of these steps done incorrectly silently degrades re-solving quality.

    R = R.cuda(); S = S.cuda(); sigma = sigma.cuda()
    infer_stream = torch.cuda.Stream()
    deadline = time.monotonic() + time_budget_sec

    while time.monotonic() < deadline:
        # Forward reach (GPU)
        pi = forward_reach_cuda(tree, sigma)

        # Async GPU leaf eval (overlap with CPU regret update from prev iter)
        leaves = get_leaves(tree)
        batch = build_features_cuda(leaves, pi, game_state)
        with torch.cuda.stream(infer_stream):
            leaf_ev = vnet.infer_fp16(batch)

        # Backward CFV (GPU, waits for leaf_ev)
        infer_stream.synchronize()
        cfv = backward_cfv_cuda(tree, leaf_ev.float(), sigma, pi)

        # Regret update + prune (GPU)
        R = torch.clamp(R + instant_regret_cuda(cfv, sigma, pi), min=0.0)
        sigma = regret_matching_cuda(R)
        S.add_(sigma)
        prune_low_reach_cuda(tree, pi, threshold=1e-5)

    return normalize_cuda(S[root_infoset])
```

---

## 14. Key Papers to Read

- **Libratus** (Brown & Sandholm 2017): blueprint + real-time re-solving architecture for heads-up
- **Pluribus** (Brown & Sandholm 2019): multiway 6-max; nested subgame solving; blueprint via MCCFR
- **DCFR** (Brown & Sandholm 2019): discounted CFR, faster convergence for blueprint training
- **DREAM** (Steinberger et al. 2020): deep regret minimization, GPU-native, value network training
- **Potential-Aware Abstraction** (Gilpin et al. 2007): card bucketing that preserves draw value
- **ReBeL** (Brown et al. 2020): recursive belief-based learning, blueprint + re-solving unified