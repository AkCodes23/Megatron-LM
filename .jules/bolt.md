## 2025-02-23 - [PyTorch Tensor Creation & Synchronization]
**Learning:** `torch.tensor(list_of_tensors)` forces data to CPU and back, and `float(cuda_tensor)` causes pipeline stalls (synchronization). These are common but silent performance killers in "glue" code.
**Action:** Use `torch.stack()` to keep tensors on GPU, and keep scalar results as 0-d tensors to avoid sync.
