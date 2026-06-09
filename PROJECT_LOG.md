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
