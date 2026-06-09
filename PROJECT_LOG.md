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
