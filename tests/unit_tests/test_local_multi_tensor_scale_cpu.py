
import pytest
import torch
import math
from megatron.core.utils import local_multi_tensor_scale

def test_local_multi_tensor_scale_cpu():
    # Test case 1: In-place scaling (Optimized path)
    tensor_size = 100
    scale = 2.0
    tensors = [torch.ones(tensor_size) for _ in range(5)]
    # Use same list for src and dst
    local_multi_tensor_scale(None, None, [tensors, tensors], scale)

    for t in tensors:
        assert torch.allclose(t, torch.ones(tensor_size) * scale), "In-place scaling failed"

    # Test case 2: Out-of-place scaling (Legacy path)
    src_tensors = [torch.ones(tensor_size) for _ in range(5)]
    dst_tensors = [torch.zeros(tensor_size) for _ in range(5)]
    scale = 0.5

    local_multi_tensor_scale(None, None, [src_tensors, dst_tensors], scale)

    for src, dst in zip(src_tensors, dst_tensors):
        assert torch.allclose(src, torch.ones(tensor_size)), "Source tensor modified in out-of-place op"
        assert torch.allclose(dst, torch.ones(tensor_size) * scale), "Out-of-place scaling failed"

    # Test case 3: Empty lists
    local_multi_tensor_scale(None, None, [[], []], scale)
    # Should not raise error

    # Test case 4: Single tensor
    single_src = [torch.tensor([10.0])]
    single_dst = [torch.tensor([0.0])]
    local_multi_tensor_scale(None, None, [single_src, single_dst], 3.0)
    assert single_dst[0].item() == 30.0

if __name__ == "__main__":
    test_local_multi_tensor_scale_cpu()
