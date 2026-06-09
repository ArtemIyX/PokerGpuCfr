# PROJECT_LOG.md

A running log of what was attempted, what changed, what worked, and what failed — so any LLM or human can continue the project without guessing.

Rule: Every task ends with a new entry. Keep it short, factual, and actionable.

---

## How to write an entry (template)

### YYYY-MM-DD HH:MM - <Short task title>
**Goal**
- What we wanted to achieve (1-3 bullets).

**Context**
- Why this was needed / where it fits (optional).
- Links to relevant docs/issues/PRs.

**Work done**
- Bullet list of concrete actions taken.
- Mention key files edited/created and what changed.

**Result**
- ✅ Success / ⚠️ Partial / ❌ Failed
- What the outcome is (1-3 bullets).

**Evidence**
- Commands run + important outputs (snippets).
- Benchmarks/profiling numbers.
- Screenshots/plots links (if any).

**Why it worked / failed**
- Root cause or key reason (not a novel, just the core).

**Follow-ups**
- Next tasks to do.
- Open questions.
- Risks / gotchas discovered.

**Artifacts**
- Files produced (models, checkpoints, logs, binaries).
- Where they are stored and how to reproduce.


---

## Entries

### 2026-06-09 11:27 - Implemented terminal detection helpers
**Goal**
- Add explicit terminal and showdown detection on top of the game state model.

**Work done**
- Added `src/pokergpu/core/terminal.py`.
- Implemented helpers for active players, non-all-in active players, terminal detection, showdown detection, and hand completion detection.
- Added tests in `tests/test_terminal.py`.
- Integrated terminal helpers into `src/pokergpu/core/transitions.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark terminal detection done.

**Result**
- ✅ Success
- Terminal and showdown logic now lives in one reusable place instead of being implicit inside transitions only.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 33 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `63 passed in 3.22s`

**Why it worked / failed**
- Extracting the logic makes payout computation and solver leaf checks simpler and less error-prone.

**Follow-ups**
- Implement payout computation.
- Then add public-state signature.

### 2026-06-09 11:24 - Implemented immutable transition engine
**Goal**
- Add action application and undo support on top of the typed game state model.

**Work done**
- Added `src/pokergpu/core/transitions.py`.
- Implemented `AppliedTransition`, `apply_action`, `apply_action_with_record`, and `undo_transition`.
- Added action application for `fold`, `check`, `call`, `bet`, and `raise`.
- Added next-player selection, showdown/terminal phase updates, and stack/commitment updates.
- Added tests in `tests/test_transitions.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark the transition engine item done.

**Result**
- ✅ Success
- Project now has immutable action transitions and a simple undo path for future traversal logic.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 31 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `60 passed in 3.76s`

**Why it worked / failed**
- Immutable transitions make correctness easier now and give us a clean base for later search and CFR traversal.

**Follow-ups**
- Implement terminal detection explicitly.
- Add payout computation.
- Add public-state signature.

### 2026-06-09 11:20 - Added typed game state model
**Goal**
- Start section 4 with an immutable typed game state model.

**Work done**
- Added `src/pokergpu/core/state.py`.
- Implemented `HandPhase`, `PlayerState`, and `GameState`.
- Linked game state to `Board`, `BettingRoundState`, and per-player hole cards/status.
- Added validation for duplicate players, duplicate cards, dealer validity, and player alignment with betting state.
- Added tests in `tests/test_state.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark the game state model item done.

**Result**
- ✅ Success
- Project now has a typed immutable hand state object ready for transitions and terminal logic.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 29 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `54 passed in 3.81s`

**Why it worked / failed**
- Keeping game state separate from transition logic makes the next action-apply layer easier to build and test.

**Follow-ups**
- Implement action apply/undo or state transition engine.
- Then add terminal detection and payout computation.

### 2026-06-09 11:17 - Added evaluator performance benchmarks
**Goal**
- Add benchmark coverage for evaluator single-hand and batch-hand performance.

**Work done**
- Added `tests/test_treys_evaluator_benchmark.py`.
- Benchmarked single 5-card evaluation.
- Benchmarked single 7-card evaluation.
- Benchmarked batch 5-card evaluation.
- Benchmarked batch 7-card evaluation.
- Updated `PLAN.md` to mark evaluator benchmarking done.

**Result**
- ✅ Success
- Evaluator layer now has baseline throughput measurements for future optimization comparisons.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 27 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `50 passed in 3.15s`
- Benchmark means from `tests/test_treys_evaluator_benchmark.py`:
  - single 5-card: ~`3.58 us`
  - single 7-card: ~`9.36 us`
  - batch 5-card (`1000` hands): ~`3278.90 us`
  - batch 7-card (`1000` hands): ~`9427.94 us`

**Why it worked / failed**
- `pytest-benchmark` gave a low-friction way to capture repeatable baseline numbers inside the existing test workflow.

**Follow-ups**
- Start section 4 game state and transitions.
- Define a typed game state model around board, betting state, and player statuses.

### 2026-06-09 11:15 - Added evaluator correctness regression tests
**Goal**
- Add broader correctness tests against known hand rankings and class names.

**Work done**
- Extended `tests/test_treys_evaluator.py`.
- Added 5-card known hand-class cases from high card through royal flush.
- Added ordering regression across the main 5-card hand categories.
- Added a 7-card test confirming the evaluator selects the best 5-card hand.
- Cleaned the mypy `treys` override in `pyproject.toml`.
- Updated `PLAN.md` to mark correctness tests done.

**Result**
- ✅ Success
- Evaluator coverage now checks both hand-class labels and score ordering behavior.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 26 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `46 passed in 0.10s`

**Why it worked / failed**
- Using fixed known hands gives stable regression coverage without depending on solver code yet.

**Follow-ups**
- Add evaluator performance benchmark.
- Then move to section 4 game state and transitions.

### 2026-06-09 11:13 - Added batch hand evaluation API
**Goal**
- Add batch 5-card and 7-card evaluation APIs on top of the typed `treys` wrapper.

**Work done**
- Extended `src/pokergpu/eval/treys_evaluator.py` with batch evaluator methods.
- Added top-level helper functions for batch 5-card and 7-card evaluation.
- Updated `src/pokergpu/eval/__init__.py` exports.
- Added batch tests in `tests/test_treys_evaluator.py`.
- Updated `PLAN.md` to mark batch evaluation API done.

**Result**
- ✅ Success
- Evaluation layer now supports both single-hand and batch-hand typed APIs.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 26 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `34 passed in 0.05s`

**Why it worked / failed**
- A simple iterable-in, tuple-out API is enough for early solver work and easy to optimize later.

**Follow-ups**
- Add broader correctness tests against known hand rankings.
- Add evaluator benchmarks.

### 2026-06-09 11:10 - Added typed treys evaluator wrapper
**Goal**
- Start section 3 hand evaluation layer.
- Wrap `treys` behind typed 5-card and 7-card evaluator APIs.

**Work done**
- Added `src/pokergpu/eval/`.
- Added `src/pokergpu/eval/treys_evaluator.py` with `TreysHandEvaluator` and `EvaluatedHand`.
- Added top-level helper functions for 5-card and 7-card evaluation.
- Added tests in `tests/test_treys_evaluator.py`.
- Added a mypy override for `treys` in `pyproject.toml`.
- Updated `PLAN.md` to mark evaluator choice, 5-card evaluation, and 7-card evaluation done.

**Result**
- ✅ Success
- Project now has a typed evaluation layer without leaking `treys` into the rest of the codebase.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 26 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `32 passed in 0.05s`

**Why it worked / failed**
- A thin wrapper gives us stable internal APIs now and leaves room to swap evaluator backends later.

**Follow-ups**
- Add batch evaluation API.
- Add broader correctness tests against known hand classes and ordering cases.

### 2026-06-09 11:06 - Implemented NLHE action legality checks
**Goal**
- Finish action legality on top of the betting-rule helpers.

**Work done**
- Added `src/pokergpu/core/legality.py`.
- Implemented legality helpers for `check`, `call`, `fold`, `bet`, and `raise`.
- Implemented `is_legal_action(...)` against the current acting player.
- Added tests in `tests/test_legality.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark action legality and rule edge-case tests done.

**Result**
- ✅ Success
- Core NLHE betting model now has both sizing-rule helpers and action legality validation.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 23 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `28 passed in 0.04s`

**Why it worked / failed**
- Keeping legality separate from rule math made the implementation small and testable.

**Follow-ups**
- Start section 3 hand evaluation layer.
- Decide whether to wrap `treys` first or use `pokerkit` only for reference tests.

### 2026-06-09 11:03 - Implemented NLHE betting rule helpers
**Goal**
- Add typed action and raise-rule helpers for no-limit hold'em.

**Work done**
- Added `src/pokergpu/core/actions.py` with `ActionType` and `Action`.
- Added `src/pokergpu/core/rules.py`.
- Implemented helpers for player stack lookup, committed chips lookup, stack after call, max raise-to, min raise increment, min raise-to, and raise bounds.
- Added tests in `tests/test_rules.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark NLHE betting rules done.

**Result**
- ✅ Success
- Project now has the rule math needed for no-limit raise sizing and stack-capped all-in handling.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 21 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `25 passed in 0.04s`

**Why it worked / failed**
- Separating rule math from action legality gives us a clear base layer that later legality checks can reuse directly.

**Follow-ups**
- Implement action legality checks next.
- Then broaden edge-case tests around folded and all-in players.

### 2026-06-09 11:00 - Implemented typed betting state models
**Goal**
- Add stack, pot, blind, and betting-round state types in `core`.

**Work done**
- Added `src/pokergpu/core/betting.py`.
- Implemented `Chips`, `PlayerIndex`, `BlindStructure`, `PlayerStack`, `Pot`, `PlayerBet`, and `BettingRoundState`.
- Added chip validation helpers and basic derived state such as `highest_bet` and `amount_to_call`.
- Added tests in `tests/test_betting.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark the betting state item done.

**Result**
- ✅ Success
- Project now has typed betting-state models ready for NLHE action rules and legality checks.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 18 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `20 passed in 0.03s`

**Why it worked / failed**
- Separating neutral betting state from action rules keeps the next legality layer simpler and easier to test.

**Follow-ups**
- Implement NLHE betting rules.
- Then implement action legality checks against this state model.

### 2026-06-09 10:56 - Implemented board representation
**Goal**
- Add typed board representation in the new `core` package.

**Work done**
- Added `src/pokergpu/core/board.py`.
- Implemented `Street` and immutable `Board`.
- Added parsing, string formatting, street detection, and validation for legal NLHE board sizes.
- Added duplicate-card protection.
- Added tests in `tests/test_board.py`.
- Updated `src/pokergpu/core/__init__.py` exports.
- Updated `PLAN.md` to mark board representation done.

**Result**
- ✅ Success
- Project now has a strict board model for preflop, flop, turn, and river states.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 16 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `15 passed in 0.02s`

**Why it worked / failed**
- Keeping board validation at the model boundary prevents illegal states from leaking into later betting and traversal logic.

**Follow-ups**
- Implement stack, pot, blind, and bet state types.
- Then add NLHE betting rules and action legality checks.

### 2026-06-09 10:53 - Moved card model into core package
**Goal**
- Start using the new subfolder structure for poker-domain code.

**Work done**
- Added `src/pokergpu/core/`.
- Moved the card-domain implementation to `src/pokergpu/core/cards.py`.
- Added `src/pokergpu/core/__init__.py`.
- Kept `src/pokergpu/cards.py` as a re-export shim for compatibility.
- Updated tests to import from `pokergpu.core.cards`.

**Result**
- ✅ Success
- Future poker-domain modules now have a clear home under `src/pokergpu/core/`.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 15 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `9 passed in 0.02s`

**Why it worked / failed**
- Moving domain code early avoids a larger refactor once more poker state modules are added.

**Follow-ups**
- Put board representation in `src/pokergpu/core/board.py`.
- Put betting state and actions in `src/pokergpu/core/`.

### 2026-06-09 10:50 - Updated SKILL.md for repo structure and typed Python workflow
**Goal**
- Align `SKILL.md` with the real project conventions.
- Document subfolder usage, current packages, and typed Python rules.

**Work done**
- Updated `SKILL.md`.
- Added project-specific implementation rules section.
- Documented that subfolders should be used under `src/pokergpu/`.
- Documented current package choices: `numpy`, `scipy`, `numba`, `torch`, `pokerkit`, `treys`, `pytest`, `ruff`, `mypy`, `pytest-benchmark`.
- Documented typed Python expectations and suggested package layout by domain.

**Result**
- ✅ Success
- `SKILL.md` now reflects the actual repo direction instead of only general solver architecture.

**Evidence**
- `SKILL.md` contains a new `0) Project-specific implementation rules` section.

**Why it worked / failed**
- The project had already converged on typed Python and package-based structure, so the doc needed to catch up.

**Follow-ups**
- Move future domain modules into subfolders such as `core/`, `eval/`, and `cfr/`.
- Keep `SKILL.md` updated when architecture decisions change.

### 2026-06-09 10:47 - Implemented typed card model and deck utilities
**Goal**
- Start section 2 core poker model.
- Add strong typed foundations for cards, ranks, suits, deck generation, and parsing.

**Work done**
- Added `src/pokergpu/cards.py`.
- Implemented `Suit` and `Rank` as `StrEnum`.
- Implemented immutable `Card` with parsing and string formatting.
- Added helpers for parsing multiple cards, formatting card lists, building a 52-card deck, and deterministic shuffling with a provided RNG.
- Added tests in `tests/test_cards.py`.
- Updated `PLAN.md` to mark the first three section 2 items done.

**Result**
- ✅ Success
- Project now has a typed card layer suitable for later board, range, and solver state code.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 13 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `9 passed in 0.02s`

**Why it worked / failed**
- Starting with immutable typed card objects gives us a clean domain base without committing to solver-specific storage yet.

**Follow-ups**
- Implement board representation next.
- Then add stack, pot, blind, and bet state types.

### 2026-06-09 10:46 - Added benchmark harness scaffold
**Goal**
- Finish section 1 of `PLAN.md`.
- Add a minimal benchmark path we can reuse for evaluator and solver performance work.

**Work done**
- Added `src/pokergpu/benchmarks.py` with typed benchmark helpers.
- Added a `benchmark` CLI mode in `src/pokergpu/cli.py`.
- Added `tests/test_benchmarks.py`.
- Updated `PLAN.md` to mark benchmark harness scaffold done.

**Result**
- ✅ Success
- Project now has a working baseline benchmark entrypoint and test coverage for it.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 11 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `3 passed in 0.01s`
- `$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pokergpu benchmark` -> prints benchmark timing output

**Why it worked / failed**
- A tiny benchmark abstraction is enough for now and keeps future performance checks consistent.

**Follow-ups**
- Start section 2: core poker model.
- First implement typed cards, suits, ranks, and deck utilities.

### 2026-06-09 10:42 - Added linting and strict typing setup
**Goal**
- Finish the tooling part of section 1.
- Set up strong Python typing and quality checks before solver code grows.

**Work done**
- Extended `pyproject.toml` with Ruff and mypy configuration.
- Enabled strict mypy checking for `src` and `tests`.
- Fixed import ordering and test typing issues so tooling passes cleanly.
- Updated `PLAN.md` to mark formatting, lint, and type-check tooling done.

**Result**
- ✅ Success
- Project now has strict static checks similar to a typed-language workflow.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m ruff check .` -> `All checks passed!`
- `.\\.venv\\Scripts\\python.exe -m mypy` -> `Success: no issues found in 9 source files`
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `2 passed in 0.01s`

**Why it worked / failed**
- Doing this early keeps domain and solver APIs precise before more modules appear.

**Follow-ups**
- Add benchmark harness scaffold.
- Then start section 2 core poker model.

### 2026-06-09 10:39 - Scaffolded Python project foundation
**Goal**
- Start section 1 of `PLAN.md`.
- Create the initial Python package, entrypoint, config, logging, and tests.

**Work done**
- Added package scaffold under `src/pokergpu`.
- Added `__main__.py`, `cli.py`, `app.py`, `config.py`, and `logging_utils.py`.
- Added `pyproject.toml` for package metadata and `tests/conftest.py` for local import path setup.
- Added `pytest.ini` and basic tests in `tests/test_config.py` and `tests/test_smoke.py`.
- Updated `PLAN.md` to mark completed foundation items.

**Result**
- ✅ Success
- Project now has a minimal runnable Python app and working test baseline.

**Evidence**
- `.\\.venv\\Scripts\\python.exe -m pytest -q` -> `2 passed in 0.02s`
- `$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pokergpu` -> `PokerGPU ready on device=auto`

**Why it worked / failed**
- A small package-first scaffold gives us a stable base for adding solver modules without reworking imports later.

**Follow-ups**
- Add formatting, lint, and type-check configuration.
- Start section 2: core poker types and NLHE rules.

### 2026-06-09 10:33 - Replaced eval7 after pip build failure
**Goal**
- Fix dependency install friction in the evaluation stack.

**Work done**
- Removed `eval7` from `requirements.txt`.
- Added `treys` as the initial hand evaluation package instead.

**Result**
- ✅ Success
- Requirements are now less likely to fail on wheel build during manual install.

**Evidence**
- Pip failed while building `eval7`.
- `requirements.txt` now uses `treys>=0.1.8`.

**Why it worked / failed**
- `eval7` can require build steps that are less convenient across environments.
- `treys` is a simpler path for early project progress.

**Follow-ups**
- Re-run `pip install -r requirements.txt`.
- If evaluator speed becomes a bottleneck later, switch to a faster backend behind an internal evaluator interface.

### 2026-06-09 10:31 - Updated Python dependencies
**Goal**
- Align `requirements.txt` with the actual project plan.
- Choose packages that speed up poker logic, evaluation, benchmarking, and future GPU work.

**Work done**
- Updated `requirements.txt`.
- Kept core numeric stack: `numpy`, `scipy`, `numba`.
- Kept `pokerkit` for rules/state support.
- Added `eval7` for poker hand evaluation.
- Added `torch` as the first practical Python GPU/value-model path.
- Added `matplotlib` and `pytest-benchmark` for reporting and measurement.

**Result**
- ✅ Success
- Dependency list is now closer to the planned architecture and implementation order.

**Evidence**
- Edited `requirements.txt` at repo root.
- Packages now cover core math, poker logic, evaluator, ML/GPU path, tests, and benchmarks.

**Why it worked / failed**
- These packages support the planned build order without forcing an early jump into a more fragile GPU stack.

**Follow-ups**
- Run `pip install -r requirements.txt`.
- If `torch` install path is inconvenient on your machine, split GPU extras into a separate requirements file later.

### 2026-06-09 10:28 - Added tracked implementation plan
**Goal**
- Create a full project plan from zero to end state.
- Make plan items easy to mark done later.

**Work done**
- Added `PLAN.md`.
- Structured the plan as phased checklists from project setup through toy CFR, GPU batching, postflop re-solving, model training, scaling, and validation.

**Result**
- ✅ Success
- Repository now has a trackable execution plan with `[ ]`, `[-]`, and `[x]` states.

**Evidence**
- `PLAN.md` exists at repo root.
- Plan includes both implementation order and future progress tracking.

**Why it worked / failed**
- The project already had architecture notes, so it was possible to convert them into an execution checklist.

**Follow-ups**
- Start phase 1 by creating the Python package layout and minimal entrypoint.
- Mark completed items in `PLAN.md` as work progresses.

### 2026-06-09 10:27 - Scanned existing GPU CFR repositories
**Goal**
- Review reference repos for existing GPU CFR implementations.
- Verify whether large GPU speedups over CPU are already demonstrated.

**Work done**
- Reviewed `DEEPFOLD-SOLVER`, `gpucfr`, and `cfrx` GitHub repositories.
- Cross-checked repo claims against linked papers and recent CFR GPU literature.

**Result**
- ✅ Success
- Confirmed GPU CFR already exists in multiple forms: production solver, CUDA research implementation, and JAX accelerator-oriented library.
- Confirmed very large speedups are reported in literature, but they depend on benchmark and architecture. "100x" is real in some comparisons, not a universal constant.

**Evidence**
- `DEEPFOLD-SOLVER` README: GPU-accelerated DCFR with CUDA backend and CPU fallback.
- `gpucfr` README: parallel CFR in C++/CUDA for NVIDIA GPUs.
- `cfrx` README: JAX CFR library focused on GPUs/TPUs.
- `GPU-Accelerated Counterfactual Regret Minimization` reports up to ~401x vs OpenSpiel Python and ~204x vs OpenSpiel C++.
- `Real-Time Parallel Counterfactual Regret Minimization` reports ~3.3-3.4x end-to-end speedup on HUNL postflop with CPU+GPU pipeline.

**Why it worked / failed**
- Repos and papers align on the same pattern: dense arrays, batching, and regularized compute are what make GPU CFR fast.

**Follow-ups**
- Inspect code structure in the three repos for reusable design patterns.
- Extract concrete implementation choices for our Python version: flat tree arrays, batching layer, and CPU/GPU split.

### 2026-06-09 10:00 - Repository initialized
**Goal**
- Create baseline repo structure for GPU-CFR NLHE project.
- Add `.gitignore`, `requirements.txt`, and project docs skeleton.

**Work done**
- Added `.gitignore` for Python + CUDA builds.
- Added `requirements.txt` with core deps.
- Added `SKILL.md` documenting intended architecture.

**Result**
- ✅ Success

**Evidence**
- `git status` shows only intended tracked files.
- `pip install -r requirements.txt` succeeds on local machine.

**Why it worked / failed**
- N/A

**Follow-ups**
- Create `src/` layout and minimal runnable entrypoint.
