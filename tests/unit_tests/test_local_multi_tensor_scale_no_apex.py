
import torch
from megatron.core.utils import local_multi_tensor_scale


def test_local_multi_tensor_scale_cpu():
    device = torch.device('cpu')
    scale = 0.5

    # Test case 1: In-place scaling
    t1 = torch.tensor([1.0, 2.0, 3.0], device=device)
    t2 = torch.tensor([4.0, 5.0], device=device)
    tensor_list = [t1, t2]
    # In-place: [tensor_list, tensor_list]
    # Note: local_multi_tensor_scale expects [src_list, dst_list]
    tensor_lists = [tensor_list, tensor_list]

    expected_t1 = t1 * scale
    expected_t2 = t2 * scale

    local_multi_tensor_scale(None, None, tensor_lists, scale)

    assert torch.allclose(t1, expected_t1), f"Expected {expected_t1}, got {t1}"
    assert torch.allclose(t2, expected_t2), f"Expected {expected_t2}, got {t2}"

    # Test case 2: Out-of-place scaling
    src1 = torch.tensor([1.0, 2.0, 3.0], device=device)
    src2 = torch.tensor([4.0, 5.0], device=device)
    src_list = [src1, src2]

    dst1 = torch.zeros_like(src1)
    dst2 = torch.zeros_like(src2)
    dst_list = [dst1, dst2]

    tensor_lists = [src_list, dst_list]

    local_multi_tensor_scale(None, None, tensor_lists, scale)

    expected_dst1 = src1 * scale
    expected_dst2 = src2 * scale

    assert torch.allclose(dst1, expected_dst1), f"Expected {expected_dst1}, got {dst1}"
    assert torch.allclose(dst2, expected_dst2), f"Expected {expected_dst2}, got {dst2}"

    # Verify src is untouched
    assert torch.allclose(src1, torch.tensor([1.0, 2.0, 3.0], device=device))

    # Test case 3: Empty lists
    # Should not crash
    local_multi_tensor_scale(None, None, [[], []], scale)

    # Test case 4: Single empty tensor
    empty_t = torch.tensor([], device=device)
    tensor_lists = [[empty_t], [empty_t]]
    local_multi_tensor_scale(None, None, tensor_lists, scale)
    assert empty_t.numel() == 0

    # Test case 5: Tensor scale (should fallback to loop but work)
    scale_tensor = torch.tensor(0.5, device=device)
    t1 = torch.tensor([10.0, 20.0], device=device)
    tensor_list = [t1]
    tensor_lists = [tensor_list, tensor_list]

    local_multi_tensor_scale(None, None, tensor_lists, scale_tensor)

    expected_t1 = torch.tensor([5.0, 10.0], device=device)
    assert torch.allclose(t1, expected_t1), f"Expected {expected_t1}, got {t1}"


if __name__ == "__main__":
    test_local_multi_tensor_scale_cpu()
