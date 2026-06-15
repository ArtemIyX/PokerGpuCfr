# CFR 7-Stage Pipeline

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

---
 
**Stage details:**
 
**S1 ForwardProfile** — infosets form independent chains; propagate reach sequentially within chain, parallel across chains. `πi(Ichild) = πi(Iparent) · σi(Iparent, a)`. Accumulate cumulative strategy π̄i(I) simultaneously.
 
**S2 AggregateProbSum** — per node: aggregate reaches into 3 levels (Psum, Pcard, Phand) for O(1) blocking correction in S3. Parallel by node × card (52 dims). Builds leaf input tensor X ∈ R^(m×d) for GPU. Applies abstraction projection if used.
 
**S3 Opponent Reach** — naive summation is O(|H|²). Instead: (1) compute per-node ratios µ[n,h] via inclusion-exclusion on S2 aggregates (∥ node), (2) propagate π−i along chains (∥ chain). Avoids quadratic cost.
 
**S4 ShowdownEquity** — rank-sorted linear scan reduces per-node cost from O(n²) to O(n). Independent nodes, trivially parallel. Blocking corrections applied.
 
**S5 BatchLeafEval** — single GPU forward pass: V̂ = fθ(X) ∈ R^(m×k). CPU-GPU transfer exactly twice per iteration. GPU time masked by S3–4 CPU branch on postflop.
 
**S6 BackwardCFV** — weight leaf predictions by π−i(I) and chance probs. Propagate bottom-up within chains in reverse topological order. `vi(σ,I) = Σa σi(I,a)·vi(σ,I·a)`. Chains independent, parallel.
 
**S7 UpdateRegret** — `ri(I,a) = vi(σ,I·a) − vi(σ,I)`. Purely local per infoset, embarrassingly parallel. Only this stage changes across CFR variants.
 

# CFR variants (iterative, fixed tree, parallelizable):

- CFR [Zinkevich et al., 2007] — regret-based self-play for imperfect-information games
- CFR+ [Tammelin, 2014] — floors negative regrets
- DCFR [Brown & Sandholm, 2019] — asymmetric discounting
- Predictive CFR+ [Farina et al., 2021] — predictive regret matching



# EV, Heuristic Eval, EV Normalization

`EV(action) = sum( P(outcome) * value(outcome) )`

EV = average chips won/lost across all possible continuations.
Solver maximizes EV at every decision node assuming opponent plays Nash.

## Heuristic Eval

Scalar approximation of EV used when exact computation is too expensive.

Inputs (weakest to strongest):
- SPR as playability proxy
- Hand strength relative to board texture

Use cases :
- **Leaf nodes**: replace terminal EV when tree is truncated
- **Action pruning**: prune actions with heuristic EV far below best candidate (MCCFR)

## EV Unit: Big Blinds (BB)

Store all EVs as BB. Pot starts at 1.5 BB (SB + BB).
All bets, raises, and EVs are multiples of 1 BB.
Report winrate as mBB/hand or BB/100.

## EV Scaling to Spot

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