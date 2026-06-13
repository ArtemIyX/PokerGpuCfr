# Single-Spot GPU CFR Speedup Plan

Goal: reduce per-iteration time for one postflop tree and raise GPU utilization from 5-10% to something close to saturation.

Current symptom:
- CPU work continues during iteration.
- GPU kernels are too small or too serialized.
- One iteration is about 1.1 s.
- CFR needs many iterations, so the solver is too slow for practical re-solving.

Main idea:
- Stop treating one iteration as a Python loop over tree work.
- Turn one iteration into a small number of large GPU passes.
- If one tree is still too small, batch more work around the tree.

## 1. Target architecture

Separate the solver into two phases:

1. Setup phase on CPU
   - Build the tree once.
   - Pack it into flat arrays.
   - Upload all static data to GPU.
   - Precompute all schedules, offsets, masks, and layouts.

2. Iteration phase on GPU
   - Forward reach pass.
   - Leaf evaluation.
   - Backward value pass.
   - Regret update.
   - Average strategy update.

After setup, CPU should only:
- launch kernels
- manage time budget
- read back final root outputs

## 2. Eliminate CPU work in the hot loop

The first bottleneck to remove is any per-node or per-infoset Python work.

Rules:
- [x] No recursion in the iteration path.
- [x] No dict lookups in the iteration path.
- [x] No per-node object creation.
- [x] No conversion between Python objects and tensors during iteration.
- No CPU normalization or card-removal loops during iteration.

All of these must be prepacked into tensors or masks before the solve starts.

## 3. Flatten the tree

Represent the tree as dense arrays:

- `node_type`
- `first_child`
- `child_count`
- `action_id`
- `infoset_id`
- `parent_id`
- `street`
- `depth`
- `terminal_payoff`
- `chance_prob`

Build a level schedule:
- [x] nodes grouped by depth
- [x] child lists stored contiguously
- [x] transpose adjacency stored too

This lets each iteration run as a small number of level-wise kernels instead of a traversal.

## 4. Move all repeated state to GPU

Resident GPU state should include:

- regret table `R`
- strategy sum `S`
- current strategy `sigma`
- reach arrays
- counterfactual value arrays
- leaf feature buffers
- masks for legal actions and card removal

If a tensor is used every iteration, it should not live on CPU.

## 5. Batch by level, not by node

The tree is small at the node level, so a kernel per node wastes GPU.

Use:
- one kernel per level
- one kernel per infoset block
- one kernel per leaf batch

This increases occupancy and reduces launch overhead.

If a level is still too small:
- merge adjacent shallow levels
- process multiple root subproblems together
- batch multiple sampled traversals of the same tree

## 6. Increase parallelism inside one spot

If one spot does not fill the GPU, the solver should manufacture more parallel work.

Good options:

1. Batch multiple chance samples.
   - Solve several public-card outcomes together.
   - Share the same tree structure where possible.

2. Batch multiple leaf evaluations.
   - Build a large leaf frontier.
   - Evaluate leaf values in one VNet call.

3. Batch multiple root action branches.
   - Run all legal root actions in parallel.
   - Propagate each branch together.

4. Batch multiple independent subtrees.
   - Same board.
   - Same abstraction.
   - Different sampled ranges or rollout seeds.

This is the main way to use a large GPU when a single tree is not enough.

## 7. Make kernels coarse

The GPU is underused because the kernels are likely too small.

What to do:
- Fuse sequential tensor ops.
- Avoid tiny elementwise kernels.
- Avoid repeated `synchronize()` calls.
- Keep forward pass, leaf eval, and backward pass asynchronous where possible.
- Use large contiguous buffers rather than many small tensors.

Kernel goal:
- fewer launches
- larger work per launch
- higher memory coalescing

## 8. Prefer approximate CFR variants if exact iteration is too slow

For a large single tree, exact full-tree CFR may still be too expensive.

Practical options:

1. CFR+ / DCFR
   - Keep if the GPU can handle full passes.
   - Best when the tree fits cleanly in VRAM.

2. MCCFR external sampling
   - Use when the tree is too large for full passes.
   - Sample fewer trajectories.
   - Batch leaf evaluation on GPU.

3. Depth-limited re-solving
   - Cut the tree at a fixed depth.
   - Use heuristic at the frontier. (In future train value network)

4. Hybrid
   - exact on shallow public tree
   - sampled on deeper branches
   - network at leaves

If 64 to 1024 full iterations are too slow, a sampled or depth-limited method is probably required.

## 9. Cache everything invariant

Cache once:
- board canonicalization
- legal action maps
- bet sizing maps
- card-removal masks
- leaf feature templates
- infoset offsets
- adjacency layouts
- action translation tables

Do not rebuild these every iteration.

## 10. Profile the real bottleneck

Measure these separately:

- tree build time
- GPU upload time
- forward reach time
- leaf eval time
- backward value time
- regret update time
- strategy normalization time
- CPU idle gaps between kernels

The goal is to find whether the bottleneck is:
- Python overhead
- kernel launch overhead
- memory bandwidth
- leaf evaluation
- sparse traversal shape

Optimize in that order.

## 11. Priority order

Do these first:

1. Make the entire iteration GPU-resident.
2. Remove Python recursion and per-node work.
3. Pack the tree into flat arrays and level schedules.
4. Batch leaf evaluation and backward propagation.
5. Merge small kernels.
6. Add micro-batching across multiple samples/subtrees.
7. Switch to MCCFR or depth-limited solving if full CFR is still too slow.

## 12. Success criteria

Good signs:
- CPU stays idle during iteration.
- GPU utilization rises well above 5-10%.
- One iteration becomes much faster than 1.1 s.
- Root strategy can be produced within the time budget.

Bad signs:
- frequent host-device transfers
- many tiny kernel launches
- repeated tensor reallocations
- per-iteration tree rebuilds
- Python loops in the hot path

## 13. Recommended immediate next step

Instrument `single_spot_postflop_solve.py` and the GPU runtime to log:
- time spent on CPU prep
- time spent in each GPU phase
- number of kernel launches per iteration
- number of leaf nodes per batch
- GPU batch size at each pass

Then decide whether to:
- fuse kernels
- batch more spots
- switch to sampled CFR
- or depth-limit the tree
