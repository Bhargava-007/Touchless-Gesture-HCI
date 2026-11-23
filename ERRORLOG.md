# Error Log

Newest entries first. Every real error or dead end encountered during development is recorded here with its exact symptom, root cause, fix, and verification evidence.

---

## 2026-11-24 — Phase 4: `torch.load(..., weights_only=True)` rejected `TorchVersion` and `np.str_` in checkpoint metadata
- **Phase:** Phase 4 (`feat/classifier`)
- **Symptom:** `_pickle.UnpicklingError: Weights only load failed ... Unsupported global: GLOBAL torch.torch_version.TorchVersion` and `numpy._core.multiarray.scalar`.
- **Root Cause:** `torch.__version__` (`TorchVersion`) and `np.random.Generator.permutation(session_ids)` (`np.str_`) are subclasses/scalars rather than plain Python primitives, so saving them inside checkpoint `metadata` caused `torch.load(..., weights_only=True)` to reject the pickle.
- **Fix / Workaround:** Cast `str(torch.__version__)` in `tghci.train`, `[str(s) for s in rng.permutation(...)]` in `tghci.dataset`, and sanitise `metadata` via `json.loads(json.dumps(metadata))` in `tghci.model.save_checkpoint`.
- **Evidence:** `.venv\Scripts\python.exe -m pytest -q` passed on both `cpu` and `cuda`.

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
