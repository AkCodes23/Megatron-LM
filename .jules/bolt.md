## 2025-10-27 - Mixed Precision Safety in `local_multi_tensor_l2_norm`
**Learning:** `torch._foreach_norm` returns a list of tensors with the same dtype as the input. When calculating L2 norms for `float16` or `bfloat16` tensors, using `_foreach_norm` directly can lead to overflow because it accumulates in the input precision.
**Action:** Only use `_foreach_norm` optimization for `float32` tensor chunks. For lower precision chunks, fall back to `torch.norm(..., dtype=torch.float32)` to ensure safe accumulation in higher precision. Do not flatten mixed-precision tensor lists into a single list for `_foreach_norm` as it will fail or be unsafe.

## 2025-10-27 - Test Environment Stability
**Learning:** `get_te_version()` returns `None` when Transformer Engine is not installed, causing `TypeError` in version checks like `is_te_min_version`.
**Action:** Always check for `None` return from `get_te_version()` before comparing versions. This is critical for running tests in environments without optional dependencies.
