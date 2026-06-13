# PokerGPU Architecture

This repository is a Python poker solver research project with four main layers:

1. `core` for game rules and state.
2. `abstraction` for hand and action compression.
3. `cfr` and `tree` for solving logic.
4. `eval`, `runtime`, and `value_network` for batched leaf evaluation and re-solving.

## Entry points

- `src/pokergpu/__main__.py`: module entry point.
- `src/pokergpu/app.py`: creates the app settings and configures logging.
- `src/pokergpu/benchmarks.py`: benchmark helper.
- `scripts/benchmark_postflop.py`: postflop throughput benchmark.
- `scripts/random_postflop_solve.py`: spot sampling and solve runner.

## Package layout

### `src/pokergpu/core/`
Game mechanics and state.

- `cards.py`: cards and deck helpers.
- `board.py`: board representation.
- `betting.py`: pot, stacks, blinds, and betting round state.
- `actions.py`: legal action model.
- `state.py`: full game state.
- `rules.py`: game rules.
- `legality.py`: action legality checks.
- `transitions.py`: state transitions.
- `terminal.py`: terminal detection.
- `payouts.py`: showdown and payout logic.
- `canonical.py`: canonicalization helpers.
- `signatures.py`: state or hand signature helpers.

### `src/pokergpu/abstraction/`
Game simplification.

- `hands.py`: hand range and hand abstractions.
- `buckets.py`: bucketing utilities.
- `actions.py`: action abstraction helpers.

### `src/pokergpu/tree/`
Public tree representation.

- `public_tree.py`: tree node and infoset structures.
- `builder.py`: builds depth-limited public trees.

### `src/pokergpu/cfr/`
Solver logic for toy and abstracted games.

- `infosets.py`: infoset bookkeeping.
- `iteration.py`: CFR iteration helpers.
- `traversal.py`: tree traversal.
- `mccfr.py`: sampling-based CFR.
- `kuhn.py`: Kuhn poker solver.
- `leduc.py`: Leduc poker solver.
- `toy_mccfr.py`: toy MCCFR experiments.
- `compare.py`: strategy or solver comparison tools.

### `src/pokergpu/eval/`
Leaf evaluation and batching.

- `interface.py`: evaluator interface.
- `backend.py`: backend selection.
- `types.py`: batch and value types.
- `cpu_stub.py`: CPU reference evaluator.
- `gpu_stub.py`: CUDA stub evaluator.
- `treys_evaluator.py`: hand strength evaluation.
- `tensor_builder.py`: tensor conversion helpers.
- `async_exec.py`: async batch execution wrapper.
- `benchmark.py`: evaluator throughput benchmark.
- `device.py`: device resolution.

### `src/pokergpu/runtime/`
Runtime solve orchestration.

- `postflop.py`: postflop resolving entry point.
- `gpu_postflop.py`: GPU-backed postflop execution.
- `gpu_compile.py`: GPU compilation helpers.
- `cache.py`: cached state and benchmark data.
- `caching.py`: cache and warm-start helpers.
- `value_network.py`: runtime value-network integration.

### `src/pokergpu/value_network/`
Value-network training and inference support.

- `model.py`: network definition.
- `dataset.py`: dataset creation.
- `target.py`: target generation.
- `equity.py`: equity helpers.
- `train.py`: training loop.
- `checkpoint.py`: checkpoint save and load.

### `src/pokergpu/benchmarks/`
Reusable benchmark modules.

- `caching_warm_start.py`: cache and warm-start benchmark.
- `cfr_threading.py`: CFR threading benchmark.

## Data flow

1. Build or load game state in `core`.
2. Apply abstraction in `abstraction`.
3. Build a public tree in `tree`.
4. Traverse with `cfr`.
5. Evaluate leaf nodes with `eval`.
6. Use `runtime` for caching, warm start, and postflop re-solving.
7. Train or query a value model from `value_network` when needed.

## Current benchmark reference

`scripts/benchmark_postflop.py` currently shows:

| Device | Depth | Nodes | Batch | Spots/sec |
|---|---:|---:|---:|---:|
| CPU | 2 | 128 | - | 151.650 |
| CPU | 2 | 512 | - | 148.701 |
| CPU | 3 | 128 | - | 30.271 |
| CPU | 3 | 512 | - | 29.354 |
| CPU | 5 | 128 | - | 60.995 |
| CPU | 5 | 512 | - | 4.494 |
| CUDA | 2 | 512 | 512 | 86.151 |
| CUDA | 2 | 512 | 4096 | 738.167 |
| CUDA | 2 | 512 | 16384 | 3057.496 |
| CUDA | 2 | 512 | 32768 | 6426.016 |
| CUDA | 2 | 512 | 65536 | 11953.981 |
| CUDA | 2 | 1024 | 512 | 91.828 |
| CUDA | 2 | 1024 | 4096 | 763.855 |
| CUDA | 2 | 1024 | 16384 | 3122.830 |
| CUDA | 2 | 1024 | 32768 | 6474.621 |
| CUDA | 2 | 1024 | 65536 | 12062.880 |
| CUDA | 3 | 512 | 512 | 26.238 |
| CUDA | 3 | 512 | 4096 | 203.675 |
| CUDA | 3 | 512 | 16384 | 837.981 |
| CUDA | 3 | 512 | 32768 | 1648.636 |

Recommended batch default: `32768`.

## Notes

- The project is still research code.
- The current solver is built around flat data, batched evaluation, and depth-limited re-solving.
- The `docs/CFR_GPU.md` file contains the longer technical guide for the GPU path.
