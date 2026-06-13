# Single-Spot Level-Batched CFR Spec

Goal: solve one public poker tree on GPU by batching nodes by depth/level, not by batching different spots.

This is the “single spot but GPU-parallel CFR” path.

## Why this path

- One spot still has many nodes.
- GPU needs large, regular work to stay busy.
- CFR has a natural level structure:
  - forward reach flows top-down
  - leaf evaluation happens at frontier nodes
  - counterfactual values flow bottom-up
  - regrets update after values
- Batch-per-level avoids launching tiny kernels per node.
- It keeps data contiguous and predictable.

## What we are solving

Input:
- one fixed `GameState`
- one public tree
- one action abstraction
- one evaluator
- one warm-start state

Output:
- root strategy
- root action EVs
- root EV
- updated regrets / strategy sums

Not part of this path:
- multi-spot batching
- tree search across multiple boards
- CPU-only fallback improvements

## Core idea

Convert the tree into flat GPU tensors.
Then process nodes level by level.

Order:
1. CPU builds the public tree once.
2. CPU canonicalizes the state and computes a cache key.
3. CPU flattens the tree into level lists.
4. GPU runs all nodes in one level together.
5. GPU stores intermediate reach and value buffers in contiguous tensors.
6. GPU updates regrets and strategy sums at infoset granularity.

## Tensor layout

Use flat, contiguous tensors only.
No pointer chasing.
No Python recursion in the hot loop.

### Per-node tensors

All indexed by `node_id`.

```text
node_type[node_id]          int8 / int32
node_depth[node_id]         int32
node_level[node_id]         int32
street[node_id]             int8 / int32
infoset_id[node_id]         int32, -1 if not a player node
first_child[node_id]        int32
child_count[node_id]        int32
is_frontier[node_id]        bool
terminal_payoff[node_id]    float32, 0 if not terminal
```

### Per-edge tensors

All indexed by `edge_id`.

```text
edge_parent[edge_id]        int32
edge_child[edge_id]         int32
edge_action_slot[edge_id]   int32
edge_chance_prob[edge_id]   float32
edge_infoset_id[edge_id]    int32, repeated from parent if relevant
edge_player[edge_id]        int8 / int32
```

### Level tensors

Each level gets a compact view.

```text
level_nodes[L]              int32[]
level_edges[L]              int32[]
level_parent_nodes[L]       int32[]
level_child_nodes[L]        int32[]
level_infoset_ids[L]        int32[]
level_action_slots[L]       int32[]
level_node_types[L]         int32[]
```

The goal is to make each level a regular GPU kernel launch.

### Reach / value buffers

```text
reach_p0[node_id]           float32
reach_p1[node_id]           float32
cfv_p0[node_id]             float32
cfv_p1[node_id]             float32
leaf_value_p0[node_id]      float32
leaf_value_p1[node_id]      float32
node_value_p0[node_id]      float32
node_value_p1[node_id]      float32
```

For multi-bucket support later:

```text
reach_p0[node_id, bucket_id]    float32
reach_p1[node_id, bucket_id]    float32
cfv_p0[node_id, bucket_id]      float32
cfv_p1[node_id, bucket_id]      float32
```

For the MVP single-spot path, keep bucket dimension out until the scalar path is stable.

### Infoset tensors

```text
infoset_offsets[infoset_id]      int32
infoset_action_count[infoset_id] int32
regret[flat_action_id]           float32
strategy_sum[flat_action_id]     float32
current_strategy[flat_action_id] float32
```

`flat_action_id = infoset_offsets[infoset_id] + action_slot`

This is the key mapping for regret updates.

## GPU execution model

### Forward pass

For each level from root to frontier:

```text
for node in level_nodes[L]:
    if chance:
        distribute reach by chance_prob
    if player node:
        propagate reach using current_strategy
    if terminal or frontier:
        stop
```

This must run as a level kernel, not per node in Python.

### Leaf evaluation

At frontier nodes:
- build a compact leaf feature batch
- run evaluator on GPU
- write values into `leaf_value_*`

For one spot, leaf evaluation is still a batch of leaves.
That batch is what makes the GPU useful.

### Backward pass

For each level from frontier to root:

```text
for node in level_nodes[L]:
    if terminal:
        node_value = terminal_payoff
    if frontier:
        node_value = leaf_eval_output
    if chance:
        node_value = sum(child_prob * child_value)
    if player:
        node_value = sum(strategy * child_value)
```

Then compute action values for each infoset.

### Regret update

For each player infoset:

```text
regret[action] += action_value[action] - infoset_value
regret = max(regret, 0)   # CFR+
strategy_sum[action] += current_strategy[action]
```

This is one parallel kernel over infosets.

## Exact data flow

### CPU side

1. Normalize public state.
2. Canonicalize board.
3. Build tree.
4. Compute cache key.
5. Flatten to arrays.
6. Group by level.
7. Upload tensors.

### GPU side

1. Initialize root reach.
2. Forward propagate reach by level.
3. Evaluate frontier batch.
4. Backward propagate values by level.
5. Update regrets.
6. Update average strategy.
7. Repeat iterations.

## Buffer ownership

Keep these resident on GPU for the whole solve:
- `node_type`
- `node_depth`
- `first_child`
- `child_count`
- `infoset_id`
- `edge_*`
- `level_*`
- `regret`
- `strategy_sum`
- `reach_*`
- `cfv_*`
- `leaf_value_*`

Upload once, reuse every iteration.

## What should be cached

Cache key should include:
- street
- canonical board
- pot
- stacks
- to-act
- dealer
- action abstraction id
- max depth
- max nodes
- evaluator id

Cache values:
- flattened tree template
- level schedule
- infoset offsets
- edge maps
- prebuilt leaf feature schema
- warm-start regrets and strategy sums

## Files to touch

### Tree and cache

- `src/pokergpu/tree/builder.py`
- `src/pokergpu/tree/public_tree.py`
- `src/pokergpu/runtime/cache.py`
- `src/pokergpu/runtime/caching.py`

### GPU solver

- `src/pokergpu/runtime/gpu_postflop.py`
- `src/pokergpu/runtime/gpu_compile.py`

### CFR core

- `src/pokergpu/cfr/traversal.py`
- `src/pokergpu/cfr/iteration.py`
- `src/pokergpu/cfr/infosets.py`

### Tests

- `tests/test_tree_builder.py`
- `tests/test_public_tree.py`
- `tests/test_caching_warm_start.py`
- `tests/test_gpu_batching.py`
- new tests for level layout and single-spot GPU path

### Script

- `scripts/predefined_flop_cuda_batch_solve.py`

## Implementation checklist

### Phase 1

- [ ] Add a flat node/edge tensor view for one tree.
- [ ] Add per-level node lists.
- [ ] Verify deterministic tree shape for identical states.
- [ ] Keep current solver working.

### Phase 2

- [ ] Move reach propagation to level kernels.
- [ ] Move backward CFV propagation to level kernels.
- [ ] Move regret update to a single infoset kernel.
- [ ] Keep warm-start load/store working.

### Phase 3

- [ ] Reduce Python overhead in tree flattening.
- [ ] Keep all hot tensors resident on GPU.
- [ ] Batch leaf evaluation into one GPU call.

### Phase 4

- [ ] Add profiling.
- [ ] Compare single-spot latency before/after.
- [ ] Compare throughput for multiple runs of the same spot.

## Rules to not break

- Do not change solver outputs when batch size is 1 unless the fix is intentional.
- Do not remove the existing CPU path until the GPU path is validated.
- Do not mutate canonical board state in place.
- Do not store raw regrets in low precision.
- Do not use Python loops for per-node hot-path work once tensors exist.
- Do not mix tree shapes in one level batch unless shapes are proven compatible.

## Success criteria

- One predefined flop spot can be solved repeatedly with warm-start reuse.
- All tree traversals on GPU are level batched.
- GPU kernels have enough work to stay busy.
- Root strategy and EV output stay stable against the current solver.
- Existing tests still pass or are updated intentionally.

## Recommended next file

Implement the flat level schedule in:
- `src/pokergpu/tree/builder.py`

Then connect it to:
- `src/pokergpu/runtime/gpu_postflop.py`

