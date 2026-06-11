# GPU-CFR Poker Solver

![Project Status](https://img.shields.io/badge/status-under%20development-orange)
![Vibe Coding](https://img.shields.io/badge/vibe--coding-yes-purple)
![AI Assisted](https://img.shields.io/badge/code%20primarily%20written%20by-AI-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![GPU](https://img.shields.io/badge/GPU-CUDA%20planned-green)
![CFR](https://img.shields.io/badge/algorithm-CFR%2B%20%2F%20DCFR-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

A Python research project for building a GPU-accelerated Counterfactual Regret Minimization solver for poker, with a long-term focus on No-Limit Texas Hold'em.

The project is not trying to exactly solve full 6-max no-limit Hold'em. That game is far too large to solve directly. Instead, this repository focuses on the practical approach used by real poker-solving systems: abstraction, flat data layouts, depth-limited solving, batched leaf evaluation, CFR variants, benchmarking, and eventually GPU-assisted re-solving.

> [!WARNING]
> This repository is under active development. APIs, package layout, solver quality, benchmarks, and GPU integration may change often. Do not treat the current code as production-ready or as a finished poker solver.

> [!NOTE]
> This is a vibe-coding / AI-assisted research project. The code is primarily written with AI help and should be reviewed carefully before serious use. Correctness, numerical stability, and performance claims must be validated with tests and reproducible benchmarks.

## Requirements

- **Core**
  - [![numpy](https://img.shields.io/badge/numpy-%3E%3D1.26-blue?logo=numpy)](https://pypi.org/project/numpy/)
  - [![scipy](https://img.shields.io/badge/scipy-%3E%3D1.11-blue?logo=scipy)](https://pypi.org/project/scipy/)
  - [![numba](https://img.shields.io/badge/numba-%3E%3D0.59-blue)](https://pypi.org/project/numba/)
  - [![tqdm](https://img.shields.io/badge/tqdm-%3E%3D4.66-blue)](https://pypi.org/project/tqdm/)
  - [![PyYAML](https://img.shields.io/badge/PyYAML-%3E%3D6.0-blue?logo=yaml)](https://pypi.org/project/PyYAML/)
  - [![orjson](https://img.shields.io/badge/orjson-%3E%3D3.10-blue)](https://pypi.org/project/orjson/)

- **Poker / evaluation**
  - [![pokerkit](https://img.shields.io/badge/pokerkit-%3E%3D0.5-purple)](https://pypi.org/project/pokerkit/)
  - [![treys](https://img.shields.io/badge/treys-%3E%3D0.1.8-purple)](https://pypi.org/project/treys/)

- **ML / GPU path**
  - [![torch](https://img.shields.io/badge/torch-%3E%3D2.3-orange?logo=pytorch)](https://pypi.org/project/torch/)

### CUDA install

For GPU support on Windows, install the CUDA wheel from PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify with:

```powershell
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

- **Data / experiments**
  - [![pandas](https://img.shields.io/badge/pandas-%3E%3D2.2-blue?logo=pandas)](https://pypi.org/project/pandas/)
  - [![matplotlib](https://img.shields.io/badge/matplotlib-%3E%3D3.8-blue)](https://pypi.org/project/matplotlib/)

- **Testing / development**
  - [![pytest](https://img.shields.io/badge/pytest-%3E%3D8.0-green?logo=pytest)](https://pypi.org/project/pytest/)
  - [![pytest-benchmark](https://img.shields.io/badge/pytest--benchmark-%3E%3D4.0-green)](https://pypi.org/project/pytest-benchmark/)
  - [![ruff](https://img.shields.io/badge/ruff-%3E%3D0.6-green?logo=ruff)](https://pypi.org/project/ruff/)
  - [![mypy](https://img.shields.io/badge/mypy-%3E%3D1.10-green)](https://pypi.org/project/mypy/)

## What this repo is

This repo is a practical implementation path toward a fast CFR-based poker solver that can support two modes:

- **Offline solving**: run many CFR iterations over an abstracted game and produce blueprint strategies, benchmark results, or training targets.
- **Real-time re-solving**: build a smaller public subtree from a current poker state and solve it within a time budget, using depth limits and leaf evaluation.

The current codebase is planned as a Python project with a clear progression from simple, testable poker logic to larger CFR systems and GPU-backed evaluation.

## Why GPU acceleration matters

CFR repeatedly traverses a game tree, computes counterfactual values, updates regrets, derives strategies through regret matching, and accumulates average strategies. The expensive parts are repeated tree evaluation and regret updates.

GPU acceleration is useful only when the workload is shaped correctly. The project is designed around GPU-friendly principles:

- flat arrays instead of pointer-heavy trees;
- dense regret and strategy tables;
- precomputed node and infoset offsets;
- large batches for leaf evaluation;
- minimal branching inside hot loops;
- CPU/GPU overlap where possible;
- optional levelized or matrix-style passes for offline solving.

The main expected GPU win is batched leaf evaluation with a learned value model. Later, parts of CFR iteration may also be expressed as dense, sparse, or block-sparse operations.

## Important reality check

Full 6-max no-limit Hold'em is astronomically large. A practical solver must use approximations:

- action abstraction, such as fixed bet-size templates;
- card abstraction or private-hand bucketing;
- public-tree depth limits;
- learned leaf value evaluation;
- pruning and caching;
- heads-up or restricted multiway prototypes before attempting larger 6-max approximations.

This project should be treated as a research and engineering framework, not as a finished commercial GTO solver.

## Current project status

The project foundation and core poker engine layers are already planned as completed in `PLAN.md`:

- package layout under `src/`;
- local runnable entrypoint;
- config system;
- logging;
- test setup;
- formatting, linting, and type-check tooling;
- benchmark harness scaffold;
- core card, deck, board, stack, pot, blind, and betting models;
- NLHE action legality checks;
- 5-card and 7-card hand evaluation;
- game-state transitions, terminal detection, and payouts;
- flat-array public-tree representation;
- baseline action abstraction.

The next major stage is card abstraction, ranges, and a tiny CFR baseline using games like Kuhn Poker and Leduc Poker before moving toward postflop Hold'em re-solving.

Actual filenames may differ, but the important design goal is separation between game rules, abstractions, tree representation, CFR logic, evaluation backends, and command-line tools.

## Solver design

The intended real-time solver loop is:

```text
for iteration in time_budget:
    compute reach probabilities through the public tree
    collect frontier / depth-limit leaves
    build a large leaf-evaluation batch
    evaluate leaves on CPU or GPU
    backpropagate counterfactual values
    update regrets using CFR+ or DCFR
    accumulate average strategy

return root strategy
```

The first solver targets should be tiny games because they are easy to verify:

1. Kuhn Poker CFR baseline.
2. Leduc Poker CFR baseline.
3. CFR+ and DCFR variants.
4. Parallel CPU traversal over flat arrays.
5. Depth-limited postflop heads-up re-solving.
6. GPU leaf-evaluation integration.
7. Learned value model.
8. Larger Hold'em abstractions.
9. Experimental multiway approximation.

## GPU strategy

The first practical GPU target is batched leaf evaluation:

1. CFR traversal reaches many depth-limit leaves.
2. Leaf features are packed into tensors.
3. A CPU stub evaluator is used first for correctness.
4. A GPU inference stub is added for integration tests.
5. A trained neural value model later replaces the stub.
6. Batch throughput, latency, and end-to-end solve speedup are benchmarked.

The project may later explore an offline “CFR as matrix operations” approach, where traversal is converted into levelized sparse or dense operations. This can provide much higher GPU utilization, but it requires more preprocessing and memory.

## Development roadmap

Short-term roadmap:

- implement private-hand indexing;
- implement range representation, normalization, and masking;
- add postflop bucket interface;
- implement dense-array infoset storage;
- implement regret matching;
- implement a basic CFR loop;
- validate convergence on Kuhn Poker and Leduc Poker.

Medium-term roadmap:

- add CFR+ and DCFR;
- separate forward pass, backward pass, and regret updates;
- add deterministic parallel CPU reductions;
- define a depth-limit policy;
- add leaf-evaluation interface;
- integrate CPU and GPU evaluator stubs.

Long-term roadmap:

- heads-up postflop re-solving prototype;
- subtree caching and warm starts;
- data generation for value-model training;
- neural value model inference;
- GPU batch throughput optimization;
- multiway approximation research;
- benchmark reports and solver-quality regression tests.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the project uses editable installation:

```bash
pip install -e .
```

## Running tests

```bash
pytest
```


## Documentation files

Recommended repository docs:

- `README.md` — high-level project overview for GitHub users.
- `PLAN.md` — task checklist and implementation roadmap.
- `CFR_GPU.md` — technical build guide for GPU-accelerated CFR.
- `PROJECT_LOG.md` — chronological task log for humans and LLM agents.


## Key implementation rules

To keep the project scalable:

- avoid hash maps in the CFR hot path;
- use flat arrays for public-tree nodes;
- store regrets and strategy sums in dense arrays;
- precompute infoset/action offsets;
- keep action abstractions small;
- batch GPU work aggressively;
- benchmark every optimization;
- validate solver correctness on toy games before scaling;
- do not claim 6-max exact solving without strong abstraction and validation.

## References and inspiration

The technical direction is based on GPU-CFR and real-time CFR research, plus existing open-source CFR and poker-solving projects:

- [CFR-Explained](https://github.com/brianberns/CFR-Explained#parallelization)
- [gpucfr](https://github.com/janrvdolf/gpucfr)
- [cfrx](https://github.com/Egiob/cfrx)
- [DEEPFOLD-SOLVER](https://github.com/a9876543245/DEEPFOLD-SOLVER)
- GPU implementation papers and real-time parallel CFR notes listed in [CFR_GPU.md](CFR_GPU.md)

## License

The project is licensed under [MIT](LICENSE)

## Disclaimer

This repository is for research and software engineering. It is under active development and should not be treated as a production-ready poker solver.

This is also a vibe-coding / AI-assisted project: code is primarily written with AI help, so every important implementation detail should be reviewed, tested, benchmarked, and validated by humans before serious use.

Poker solvers are complex, approximate systems, and performance claims must be proven with reproducible benchmarks. GPU acceleration can provide large speedups only when the workload is batched, memory-friendly, and correctly validated.
