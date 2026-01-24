import torch
import pytest
from megatron.core.utils import local_multi_tensor_l2_norm

def test_local_multi_tensor_l2_norm_correctness():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    device = "cuda"

    # Create some tensors
    tensors = [torch.rand(10, 10, device=device) for _ in range(5)]
    tensor_lists = [tensors]

    # Calculate expected norm manually
    # Sqrt(Sum(norm(t)^2))
    expected_norm_sq = 0.0
    for t in tensors:
        expected_norm_sq += torch.norm(t).item() ** 2
    expected_norm = torch.sqrt(torch.tensor(expected_norm_sq))

    # Call local_multi_tensor_l2_norm
    # args: chunk_size, noop_flag, tensor_lists, per_tensor
    result, _ = local_multi_tensor_l2_norm(None, None, tensor_lists, None)

    # Verify result
    assert result.device.type == "cuda"
    assert result.shape == (1,)
    torch.testing.assert_close(result.item(), expected_norm.item())

def test_local_multi_tensor_l2_norm_empty():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensor_lists = [[]]
    result, _ = local_multi_tensor_l2_norm(None, None, tensor_lists, None)
    assert result.item() == 0.0
