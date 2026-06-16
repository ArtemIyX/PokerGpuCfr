# Real-Time Parallel Counterfactual Regret Minimization

We present Parallel CFR, the first parallelization framework for real-time depth-limited CFR solving that seamlessly integrates pruning, abstraction, and advanced CFR variants. We decompose<br>
each CFR iteration into a pipeline of seven stages and identify two orthogonal dimensions of<br>
parallelism: **by information set and by tree node**

## Worskspace rules

> Goal of project: Runtime (GPU) Poker Solver

### Main working rules

- Keep hot paths parallelized and vectorized in place; prefer NumPy, Numba, Triton, or Torch kernels over Python loops or wrapper objects in performance-critical code.
- Try to not use tupples, nested tuples/lists, nested python nodes
- Try not to use 'Any' (mypy errors)
- Divide tasks on small subtasks that can be solved one by one
- Create short plans for each large task/problem, before solving it
- Do not revert back, mask off, turn off if something broken - fix it instead
- Do not gues problem, do not chase problem by guessing, write small tests to find exact broken part of code
- Keep files small, do not create large python files, divide it into small helpers
- Do not call python, pytest, mypy, ruff, py compile etc without permission given
- Use static typing (mypy package)
- Create tests for new features, for each found bug create test, do not break tests
- Make small python files in scripts/ if we need to check something, that tests can not cover
- Use Triton if possible, do not use .cpp(.cu) CUDA extensions, use python
- Lower count of ``cudaLaucnhKernel`` if possible
- Lower using of aten::index, aten:: etc (batch, fuse, group)

### PyTorch / Triton Optimization Rules

- Do not introduce Python-layer conversions or objects in hot paths if the same work can stay in NumPy, Numba, Torch, or Triton.
Constraints: no dynamic control flow, no Python, fixed tensor shapes/addresses.
- Pin tensors to CUDA at load time,  Never ``.item()`` in hot path — forces sync
- ``torch.compile``, Fuses ops, reduces kernel launches. Use `fullgraph=True` if no graph breaks.
- inference_mode over no_grad: faster than `no_grad`, disables version tracking entirely.
- Contiguous memory before Triton kernels, ``.contiguous()`` avoids stride penalties
- Fuse ops in Triton. One kernel for `relu -> norm -> scale` eliminates 3 allocations and 3 kernel launches.
- tl.constexpr for block sizes, enables compile-time unrolling and specialization.
- autocast (bf16/fp16), Halves memory bandwidth. BF16 preferred on Ampere+ (no overflow risk vs FP16).
- Avoid dynamic shapes. Triggers recompilation in `torch.compile` and Triton. Pad to fixed sizes or declare explicitly:
- Profile before optimizing
- CUDA Graphs + torch.compile (combined)
- Eliminate CPU-side kernel launch overhead. Best for fixed-shape inference loops.

Use static input buffer: `x.copy_(new_data)` before `g.replay()`.
```python
# Warmup
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        out = model(x)
torch.cuda.current_stream().wait_stream(s)
 
# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(x)
 
# Replay (zero kernel launch overhead)
g.replay()
```

### Game Rules

Algorithm should work in any given game:

- Leduc poker
- Kuhhn poker
- Holdem no limit heads up
- Holdem no limit 6max (in future)


## Abstraction Layer

### `hands.py`
- `PrivateHand` — two distinct cards in canonical order (lower index first)
- `RangeVector` — float32 array of length 1326 (all hole card combos), one weight per hand
- `all_private_hands()` — ordered tuple of all 1326 hands
- `private_hand_index(c1, c2)` — index into RangeVector
- `private_hand_mask(dead_cards)` — bool array, False where hand overlaps dead cards
- `RangeVector.uniform()` / `.zeros()` / `.from_values()`
- `.masked(dead_cards)` / `.normalized()` / `.normalized_masked(dead_cards)` — standard range ops

### `buckets.py`
- `StrengthTierBucketer` — maps postflop hand+board → one of 9 strength buckets (0=worst, 8=best)
- `bucket_mask(board)` → int32 array[1326], -1 if hand blocked by board
- `bucketed_range(range_vector, board)` → float32 array[9], sums range weight per bucket
- Uses `TreysHandEvaluator` internally (auto-created if not passed)

### `actions.py`
- `AbstractionProfile` — per-street bet sizes (as BB multiples) and raise-to multipliers
- `BaselineActionAbstraction` — implements `legal_actions(state) → tuple[Action, ...]`
- `make_default_profile()` — 2 bet sizes + 2–3 raise sizes per street
- `make_compact_profile()` — 1 bet size + 1 raise size per street (fast/small tree)

### When to use what
 
| Goal | Use |
|------|-----|
| Track villain range | `RangeVector`, update weights by hand, call `.normalized_masked(board_cards)` each street |
| Compress range for CFR node | `StrengthTierBucketer.bucketed_range(range, board)` |
| Enumerate actions at a node | `BaselineActionAbstraction(profile).legal_actions(state)` |
| Small/fast game tree | `make_compact_profile()` |
| Full blueprint tree | `make_default_profile()` |
| Check if hand is live | `private_hand_mask(dead_cards)` or `RangeVector.masked()` |
 
### Conventions
- `PrivateHand` always has `first < second` by deck index — use `PrivateHand.from_cards()` to avoid ordering errors
- `RangeVector` values are weights (not probabilities); call `.normalized()` before using as a distribution
- Bucket 8 = best (straight flush), bucket 0 = high card
- `StrengthTierBucketer` raises on preflop boards — postflop only

## CFR 7-Stage Pipeline

Each CFR iteration: (1) forward reach probabilities, (2) leaf/terminal eval, (3) backward CFV + regret update.

Decomposed into 7 stages, each parallelizable:

| Stage | Operation | Parallel Dim | Data Flow |
|-------|-----------|--------------|-----------|
| 1 | ForwardProfile | Information set | σ → πi, π−i |
| 2 | AggregateProbSum | Node × Card | πi → Psum[n], xleaf |
| 3 | Compute π−i(I) | Node + Infoset | Psum → π−i(I) |
| 4 | ShowdownEquity | Node | π−i(I) → vsd |
| 5 | BatchLeafEval | GPU batch | xleaf → v̂leaf |
| 6 | BackwardCFV | Node + Infoset | v̂leaf, vsd → vi(I) |
| 7 | UpdateRegret | Information set | vi(I) → Ri(I, a) |

**Execution order:**
- Stages 1–2: forward pass (root → leaves)

> Vectorized over hands per infoset (shared strategy within infoset)<br>
> Card blocking: players can't share cards, so hands face different feasible opponent sets<br>
> Requires aggregation structures for blocking correction in Stages 3+

- Stages 3–4 (CPU) and Stage 5 (GPU): parallel, no shared dependency
- Stages 6–7: backward pass, starts after both branches complete

**Properties:**
- Each stage has a defined interface and can be optimized independently
- CFR variant only affects Stage 7; Stages 1–6 are identical across variants

## CFR iteration pipeline

**ForwardProfile** *(infoset parallel)*
↓
**AggregateProbability** *(node parallel)*
↓ ↓ (parallel)

| Left branch | Right branch |
|---|---|
| **OpponentReach** *(node parallel)* | **GPULeafEval** *(GPU batch)* |
| **ShowdownEquity** *(node parallel)* | |

↓ (join)
**BackwardCFV** *(node parallel)*
↓
**UpdateRegret** *(infoset parallel)*

> The parallel block runs OpponentReach + ShowdownEquity (CPU) alongside GPULeafEval (GPU batch) concurrently.

---
 
**Stage details:**
 
**S1 ForwardProfile** — infosets form independent chains; propagate reach sequentially within chain, parallel across chains. `πi(Ichild) = πi(Iparent) · σi(Iparent, a)`. Accumulate cumulative strategy π̄i(I) simultaneously.
Information sets for a given player form<br>
disjoint chains in the game tree. Within each chain, reach probabilities propagate sequentially:<br>
πi(Ichild) = πi(Iparent) · σi(Iparent, a), (5)<br>
where a is the action leading to Ichild. Chains are mutually independent, enabling parallel processing<br>
across all chains. The cumulative reach-weighted strategy ¯πi(I) is accumulated simultaneously.<br>
 
**S2 AggregateProbSum** — per node: aggregate reaches into 3 levels (Psum, Pcard, Phand) for O(1) blocking correction in S3. Parallel by node × card (52 dims). Builds leaf input tensor X ∈ R^(m×d) for GPU. Applies abstraction projection if used.

For each node n, we aggregate reach
probabilities into three levels (total reach Psum, per-card reach Pcard, and per-hand reach Phand)<br>
that enable O(1) card-blocking corrections in Stage 3 (see Appendix ?? for definitions). Node-level<br>
aggregation is embarrassingly parallel; per-card sums add a second parallel dimension of size |C|<br>
(e.g., 52 in poker).<br>

For depth-limited solving, this stage also constructs the batched input tensor X ∈ R m×d for<br>
all m non-terminal leaf nodes, preparing them for GPU evaluation in Stage 5. When information<br>
abstraction is used, projection matrices bridge the game tree and network representation space

After the forward pass, the pipeline forks into two independent branches: Stages 3–4 on CPU and<br>
Stage 5 on GPU. Since they share no data dependency, they execute in parallel. In practice, GPU<br>
evaluation time is entirely masked by the longer CPU branch on postflop streets.

**S3 Opponent Reach** — naive summation is O(|H|²). Instead: (1) compute per-node ratios µ[n,h] via inclusion-exclusion on S2 aggregates (∥ node), (2) propagate π−i along chains (∥ chain). Avoids quadratic cost.

Сomputing the opponent counterfactual<br>
reach π−i(I) naively requires summing over all opponent hands at each information set, yielding<br>
O(|H|2) complexity. We decompose this into two phases: first, compute per-node multiplicative ratios µ[n, h] using the three-level aggregates from Stage 2 via an inclusion-exclusion formula (parallel<br>
by node); then propagate π−i along information set chains using these ratios (parallel by chain).<br>
This avoids the quadratic cost while exposing parallelism in both phases.<br>
 
**S4 ShowdownEquity** — rank-sorted linear scan reduces per-node cost from O(n²) to O(n). Independent nodes, trivially parallel. Blocking corrections applied.

At showdown terminals, each hand’s equity must be<br>
computed against the opponent’s reach-weighted range. Johanson et al. [2011] observed that hand<br>
ranks induce a total order at showdown, enabling a rank-sorted linear scan that reduces per-node<br>
complexity from O(n2) to O(n) where n is the number of hands. We adopt this technique with<br>
our card-blocking corrections.Showdown nodes are independent, so this stage parallelizes trivially<br>
across nodes. For games lacking the rank-monotonicity property of poker, one may pre-store a<br>
payoff matrix and use sparsification methods [Farina and Sandholm, 2022] to compute showdown<br>
values efficiently.

 
**S5 BatchLeafEval** — single GPU forward pass: V̂ = fθ(X) ∈ R^(m×k). CPU-GPU transfer exactly twice per iteration. GPU time masked by S3–4 CPU branch on postflop.

A single GPU forward pass evaluates all non-terminal leaves<br>
simultaneously: Vˆ = fθ(X) ∈ R m×k<br>
, where k is the output dimension (hands × players). This<br>
reduces kernel launch overhead and keeps the GPU well-utilized. CPU–GPU data transfer occurs<br>
exactly twice per iteration (input upload and output download), amortized across all leaf nodes.

 
**S6 BackwardCFV** — weight leaf predictions by π−i(I) and chance probs. Propagate bottom-up within chains in reverse topological order. `vi(σ,I) = Σa σi(I,a)·vi(σ,I·a)`. Chains independent, parallel.
 
**S7 UpdateRegret** — `ri(I,a) = vi(σ,I·a) − vi(σ,I)`. Purely local per infoset, embarrassingly parallel. Only this stage changes across CFR variants.
 

## CFR variants (iterative, fixed tree, parallelizable):

- CFR [Zinkevich et al., 2007] — regret-based self-play for imperfect-information games
- CFR+ [Tammelin, 2014] — floors negative regrets
- DCFR [Brown & Sandholm, 2019] — asymmetric discounting
- Predictive CFR+ [Farina et al., 2021] — predictive regret matching


## Integration with Pruning and Abstraction
> Online Pruning. 

Our pipeline integrates with online pruning methods that remove dominated<br>
actions before CFR iterations begin. We adopt the EVPA framework [Li and Huang, 2025], which<br>
uses a neural network ensemble to bound counterfactual values and prune actions whose optimistic<br>
value is dominated by a sibling’s pessimistic value. Li and Huang [2025] proved that this pruning is<br>
sound and demonstrated up to two orders of magnitude speedup. Since pruning is performed once<br>
as a preprocessing step, it reduces the effective tree size by a factor ρ ∈ [0, 1] for all subsequent<br>
CFR iterations, providing multiplicative speedup to every pipeline stage. Pruning decisions at<br>
each information set are independent, so the pruning pass itself is embarrassingly parallel. All<br>
experiments in this paper enable pruning by default.

## EV, Heuristic Eval, EV Normalization

`EV(action) = sum( P(outcome) * value(outcome) )`

EV = average chips won/lost across all possible continuations.
Solver maximizes EV at every decision node assuming opponent plays Nash.

### Heuristic Eval

Scalar approximation of EV used when exact computation is too expensive.

Inputs (weakest to strongest):
- SPR as playability proxy
- Hand strength relative to board texture

Use cases :
- **Leaf nodes**: replace terminal EV when tree is truncated
- **Action pruning**: prune actions with heuristic EV far below best candidate (MCCFR)

### EV Unit: Big Blinds (BB)

Store all EVs as BB. Pot starts at 1.5 BB (SB + BB).
All bets, raises, and EVs are multiples of 1 BB.
Report winrate as mBB/hand or BB/100.

### EV Scaling to Spot

Always compare EVs relative to pot size, not absolute:

```
normalized_EV = EV / pot_size
regret_threshold = epsilon * pot_size
```

Rules:
- Leaf node heuristics must be in BB, then divided by pot at that node
- Regret thresholds scale with pot, not with stack
- Subgame re-solving: all EVs relative to pot at subgame root, not full stack
- Value network training targets: normalize to BB (not raw chips) so the network generalizes across stack depths
