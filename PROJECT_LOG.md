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
