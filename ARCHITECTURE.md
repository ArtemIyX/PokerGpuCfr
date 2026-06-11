# PokerGPU Architecture Overview

This document provides a high-level architectural guide for the GPU-Accelerated Counterfactual Regret Minimization (CFR) Poker Solver project, designed to help new contributors understand the codebase structure and core concepts.

## 🎯 Project Goal
The primary objective is to build a scalable framework for solving poker game states using CFR algorithms, with a long-term focus on leveraging GPU acceleration for performance-critical tasks like leaf evaluation. The system focuses on *abstraction* (e.g., fixed bet sizes, limited depth) rather than solving full, complex games exactly.

## 📁 Core Packages Structure
The codebase is organized into several specialized modules:

### `src/pokergpu/` (Main Package Entry Point)
*   **`app.py`**: Initializes the application environment (`create_app`). It loads settings and sets up logging.
    *   *Usage:* The main initialization point for the solver logic.
*   **`config.py`**: Handles configuration loading using `Settings` dataclass, reading from environment variables (e.g., `POKERGPU_LOG_LEVEL`, `POKERGPU_DEVICE`, depth and pruning limits).
*   **`logging_utils.py`**: Provides standardized logging configuration (`configure_logging`).
*   **`__init__.py`**: Exposes the primary application factory function: `create_app`.

### `src/pokergpu/core/` (Game Rules and State Management)
This package contains the fundamental game mechanics, independent of solver logic.
*   **`cards.py`**: Defines the core representation for poker cards (`Card`, `Suit`, `Rank`). Provides utility functions to create decks, shuffle, and format card strings.
    *   *Key Classes:* `Card`.
*   **`state.py`**: Manages the entire game state across a single hand. It is the central data structure linking all components.
    *   *Key Classes:* `GameState`, `PlayerState`.
    *   *Functionality:* Tracks board cards, player hole cards, folding status, and betting round progression. Ensures card uniqueness and structural integrity.
*   **`betting.py`**: Manages the monetary aspects of the game (chips, pots, etc.).
*   **`board.py`**: Models the community cards (`Board`) and tracks the current betting street (e.g., Flop, Turn, River).
*   **`rules.py`, `payouts.py`, `legality.py`, `transitions.py`**: Enforce game rules: calculating payouts, checking action legality given a state, defining valid transitions between states, and enforcing the betting structure.

### `src/pokergpu/abstraction/` (Game Simplification)
This package abstracts complex poker reality into solvable formats for CFR.
*   **`hands.py`**: Handles the representation of hand strengths (e.g., 5-card, 7-card).
*   **`buckets.py`**: Implements card or range bucketing techniques to reduce game complexity while maintaining accuracy approximation.
*   **`actions.py`**: Defines and manages action space abstraction (e.g., standard bet sizes rather than continuous betting ranges).

### `src/pokergpu/tree/` (Search Structure)
Responsible for mapping the sequence of decisions into a searchable graph structure.
*   **`public_tree.py`**: Implements the representation of the game tree used during solving, linking states and actions together (`NodeId`, `InfosetId`).
*   **`builder.py`**: Builds depth-limited public trees and marks frontier nodes for evaluation.

### `src/pokergpu/eval/` (Leaf Evaluation)
Handles batched leaf-value evaluation for frontier nodes.
*   **`types.py`**: Defines leaf feature and value batch containers.
*   **`cpu_stub.py`**: Deterministic CPU evaluator used for local development and parity checks.
*   **`gpu_stub.py`**: CUDA-backed evaluator stub using PyTorch.
*   **`device.py`**: Resolves CPU or CUDA execution mode.
*   **`tensor_builder.py`**: Converts leaf batches into device tensors.
*   **`async_exec.py`**: Async-ready wrapper for overlapped batch execution.
*   **`benchmark.py`**: Measures leaf batch throughput.

## 🔄 Data Flow: Solver Lifecycle
The typical solve process follows this flow:

1.  **Setup/Configuration (`cli.py`):** The command line interface calls the system, loading settings and defining the target game (e.g., Kuhn Poker, Leduc Poker).
2.  **State Initialization:** A starting `GameState` is created using the rules defined in `core/`.
3.  **Tree Traversal (CFR Loop):** The solver traverses the public tree structure (`tree/public_tree.py`) based on current probabilities and action legality checks from `core/rules.py`.
4.  **Leaf Evaluation:** When a leaf node (terminal state or depth limit) is reached, the system must determine the expected value. This uses:
    *   `tree/builder.py` frontier flags to mark nodes that need evaluation.
    *   `eval/` batch builders and evaluator backends.
    *   CPU or CUDA execution depending on device selection.
5.  **Backpropagation & Update:** The calculated values are back-propagated up the tree to update regrets and average strategies, following standard CFR algorithms implemented in `cli.py` (via functions like `train_kuhn_cfr`).

## ✨ Novice Getting Started Guide

### 1. Prerequisites
*   Python 3.10+
*   Required dependencies: Check `requirements.txt`. Key libraries include `numpy`, `scipy`, `numba`, `torch` (for GPU path), and `treys`.

### 2. Setup & Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```
*Note: The project uses a modular structure; ensure all dependencies listed in `requirements.txt` are installed.*

### 3. Running Initial Tests (Recommended)
The system relies heavily on testing. Start by running the built-in test suite to confirm component isolation and correctness.
```bash
pytest
```
*Self-correction: If a specific feature is being worked on, e.g., Leduc Poker, use `pytest tests/test_leduc.py`.*

### 4. Running Solver Benchmarks (Quick Test)
To run a basic performance benchmark of the current solver iteration:
```bash
python src/pokergpu/cli.py benchmark
# Output format example: benchmark=noop iterations=1000 seconds=X.XXXXXX per_iter=Y.YYYYYY
```

For leaf-evaluation comparisons:
```bash
pytest --run-benchmarks -s tests/test_leaf_evaluation_benchmark.py
```
This prints CPU single-thread, CPU multi-thread, and CUDA timings.

### 5. Solving Toy Games (First Major Implementation)
The most stable and recommended entry point is solving toy poker games, which validate the core CFR logic on simple, bounded spaces.

**A. Kuhn Poker:** Trains the solver for a specific set of iterations and variants.
```bash
# Usage: python src/pokergpu/cli.py kuhn <iterations> --variant <VANILLA|CFR_PLUS|DCFR> [VARIANTS...]
python src/pokergpu/cli.py kuhn 2000 --variant VANILLA
```

**B. Leduc Poker:** Trains the solver for a specific set of iterations and variants.
```bash
# Usage: python src/pokergpu/cli.py leduc <iterations> --variant <VANILLA|CFR_PLUS|DCFR> [VARIANTS...]
python src/pokergpu/cli.py leduc 800 --variant CFR_PLUS
```

## ⚠️ Critical Rules for Development
1.  **Performance:** Avoid hash maps and pointer-heavy structures in the hot paths (CFR loop). Use flat arrays wherever possible.
2.  **Abstraction:** Always work with the abstracted, simplified game model until validation on real-world complexity is achieved.
3.  **Verification:** Every optimization or feature implementation *must* be accompanied by a unit test and performance benchmark before integration.
