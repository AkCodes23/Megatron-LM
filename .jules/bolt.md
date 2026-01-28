## 2024-05-20 - [PyTorch Training Loop Optimization]
**Learning:** Extracting inner functions defined inside hot loops (like training steps or FLOPs calculation) to the module level significantly reduces function creation overhead, especially in high-frequency calls.
**Action:** Always check frequently called functions for inner function definitions that don't rely on closure over changing variables.
## 2025-10-27 - Test Environment Restrictions
**Learning:** Unit tests try to download data to `/opt/data` which is read-only in sandbox.
**Action:** Use `UNIT_TEST_DATA_DIR` env var or patch `conftest.py` to redirect to a writable path like `/tmp/data`.

## 2025-10-27 - CPU Testing
**Learning:** The codebase defaults to NCCL and CUDA for distributed init, causing tests to fail on CPU-only environments.
**Action:** Patch `tests/unit_tests/test_utilities.py` to use `gloo` backend if CUDA is unavailable.

## 2025-10-28 - Corrupted Code Blocks
**Learning:** Large Python files with repetitive patterns (like FLOPs calculations for different architectures) are prone to bad merges or copy-paste errors, leading to corrupted blocks where function definitions are interrupted or duplicated.
**Action:** When seeing `SyntaxError` in the middle of a function definition, check for cut-off implementations and nested duplicated logic. Use `sed` or `replace_with_git_merge_diff` to surgically remove the garbage while preserving the intended logic (often found inside the "garbage" block as inner functions).
