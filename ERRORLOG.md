# Error Log

Newest entries first. Every real error or dead end encountered during development is recorded here with its exact symptom, root cause, fix, and verification evidence.

---

## 2026-11-24 — Phase 7: `ruff check .` failed with `I001` in `tests\test_bench_eval.py`
- **Phase:** Phase 7 (`feat/bench-eval`)
- **Symptom:** `I001 [*] Import block is un-sorted or un-formatted` at `tests\test_bench_eval.py:3:1`.
- **Root Cause:** Local test import `from tests.test_classifier import _write_synthetic_dataset` was placed after `tghci` imports.
- **Fix / Workaround:** Sorted imports in `tests/test_bench_eval.py`.
- **Evidence:** `.venv\Scripts\python.exe -m ruff check .` printed `All checks passed!`.

## 2026-11-24 — Phase 6: `ruff check .` failed with `I001` in `tests\test_app.py`
- **Phase:** Phase 6 (`feat/app`)
- **Symptom:** `I001 [*] Import block is un-sorted or un-formatted` at `tests\test_app.py:3:1`.
- **Root Cause:** Local test import `from tests.test_swipe_actions import FakeBackend` was placed after `tghci` imports.
- **Fix / Workaround:** Sorted imports in `tests/test_app.py`.
- **Evidence:** `.venv\Scripts\python.exe -m ruff check .` printed `All checks passed!`.

## 2026-11-24 — Phase 1: `pip install mediapipe` failed on DNS resolution inside sandbox
- **Phase:** Phase 1 (`chore/scaffold`)
- **Symptom:** `Failed to resolve 'pypi.org' ([Errno 11001] getaddrinfo failed)` and `ERROR: No matching distribution found for mediapipe`.
- **Root Cause:** Sandbox network restriction blocked PyPI DNS resolution (not a missing wheel).
- **Fix / Workaround:** Package installs were re-run outside the sandbox (`BypassSandbox: true`), which installed `mediapipe 1.2.3` on Python `3.13.16`.
- **Evidence:** `.venv\Scripts\python.exe -c "import mediapipe as mp; print(mp.__version__, hasattr(mp,'solutions'))"` printed `1.2.3 False`.

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
