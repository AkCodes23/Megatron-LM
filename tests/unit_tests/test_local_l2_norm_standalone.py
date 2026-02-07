
import pytest
import torch
from megatron.core.utils import local_multi_tensor_l2_norm

def test_local_multi_tensor_l2_norm_float32():
    t1 = torch.tensor([3.0, 4.0], dtype=torch.float32)
    t2 = torch.tensor([6.0, 8.0], dtype=torch.float32)
    tensor_lists = [[t1], [t2]]

    # Expected: sqrt(3^2 + 4^2 + 6^2 + 8^2) = sqrt(25 + 100) = 11.1803
    result, _ = local_multi_tensor_l2_norm(0, False, tensor_lists, False)

    assert torch.isclose(result, torch.tensor(11.1803398), atol=1e-4)
    assert result.dtype == torch.float32

def test_local_multi_tensor_l2_norm_float16_fallback():
    # Test that float16 inputs are handled correctly (falling back to torch.norm loop)
    t1 = torch.tensor([3.0, 4.0], dtype=torch.float16)
    t2 = torch.tensor([6.0, 8.0], dtype=torch.float16)
    tensor_lists = [[t1], [t2]]

    result, _ = local_multi_tensor_l2_norm(0, False, tensor_lists, False)

    assert torch.isclose(result, torch.tensor(11.1803398), atol=1e-3)
    # The result should be float32 because we cast to float32 during accumulation
    assert result.dtype == torch.float32

def test_local_multi_tensor_l2_norm_empty():
    tensor_lists = []
    result, _ = local_multi_tensor_l2_norm(0, False, tensor_lists, False)
    assert result.item() == 0.0

    tensor_lists = [[], []]
    result, _ = local_multi_tensor_l2_norm(0, False, tensor_lists, False)
    assert result.item() == 0.0

def test_local_multi_tensor_l2_norm_foreach_norm_path(mocker):
    # Mock hasattr to ensure we take the _foreach_norm path
    # But only if dtype is float32

    if not hasattr(torch, "_foreach_norm"):
        pytest.skip("torch._foreach_norm not available")

    t1 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    tensor_lists = [[t1]]

    spy = mocker.spy(torch, "_foreach_norm")

    local_multi_tensor_l2_norm(0, False, tensor_lists, False)

    spy.assert_called_once()

def test_local_multi_tensor_l2_norm_no_foreach_for_fp16(mocker):
    if not hasattr(torch, "_foreach_norm"):
        pytest.skip("torch._foreach_norm not available")

    t1 = torch.tensor([1.0, 1.0], dtype=torch.float16)
    tensor_lists = [[t1]]

    spy = mocker.spy(torch, "_foreach_norm")

    local_multi_tensor_l2_norm(0, False, tensor_lists, False)

    # Should NOT be called because dtype is not float32
    spy.assert_not_called()
