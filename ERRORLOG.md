# Error Log

Newest entries first. Every real error or dead end encountered during development is recorded here with its exact symptom, root cause, fix, and verification evidence.

---

## 2026-11-24 — Phase 0: `git -C .. status` access error inside IDE sandbox
- **Phase:** Phase 0 (Probe)
- **Symptom:** `fatal: error reading 'C:/Users/bharg/Downloads/College/DL-AC-2-2/.git'`
- **Root Cause:** The parent repository `.git` metadata sits outside the workspace sandbox boundary and is inaccessible from the sandboxed shell.
- **Fix / Workaround:** Treated as non-blocking per user amendment; skipped parent git commands and kept all git operations strictly inside `Project-TG-HCI`.
- **Evidence:** `git status` inside `Project-TG-HCI` exited 0 (`nothing to commit, working tree clean`).

## 2026-11-24 — Phase 0: `py -0p` reports no installed Pythons
- **Phase:** Phase 0 (Probe)
- **Symptom:** `No installed Pythons found!`
- **Root Cause:** The Windows `py` launcher did not see the per-user Python installations in this environment.
- **Fix / Workaround:** Invoked Python directly by its full path `C:\Users\bharg\AppData\Local\Programs\Python\Python313\python.exe` (`Python 3.13.16`).
- **Evidence:** `& "C:\Users\bharg\AppData\Local\Programs\Python\Python313\python.exe" --version` printed `Python 3.13.16`.
