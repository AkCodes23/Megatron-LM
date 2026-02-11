# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import pytest
from megatron.core.utils import local_multi_tensor_scale

def test_local_multi_tensor_scale_inplace():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tensors = 5
    tensor_size = 128
    scale = 0.5

    # Create tensors
    src_list = [torch.randn(tensor_size, device=device) for _ in range(num_tensors)]

    # Expected results (naive calculation)
    expected_list = [t * scale for t in src_list]

    # Call local_multi_tensor_scale (in-place)
    # Note: local_multi_tensor_scale expects tensor_lists = [src_list, dst_list]
    # Here src_list is dst_list
    local_multi_tensor_scale(None, None, [src_list, src_list], scale)

    # Verify correctness
    for t, expected in zip(src_list, expected_list):
        torch.testing.assert_close(t, expected)

def test_local_multi_tensor_scale_copy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tensors = 5
    tensor_size = 128
    scale = 0.5

    # Create tensors
    src_list = [torch.randn(tensor_size, device=device) for _ in range(num_tensors)]
    dst_list = [torch.randn(tensor_size, device=device) for _ in range(num_tensors)]

    # Expected results (naive calculation)
    expected_list = [t * scale for t in src_list]

    # Call local_multi_tensor_scale (copy + scale)
    local_multi_tensor_scale(None, None, [src_list, dst_list], scale)

    # Verify correctness
    for t, expected in zip(dst_list, expected_list):
        torch.testing.assert_close(t, expected)

def test_local_multi_tensor_scale_copy_scale_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tensors = 5
    tensor_size = 128
    scale = 1.0

    # Create tensors
    src_list = [torch.randn(tensor_size, device=device) for _ in range(num_tensors)]
    dst_list = [torch.randn(tensor_size, device=device) for _ in range(num_tensors)]

    # Expected results (naive calculation)
    expected_list = [t * scale for t in src_list]

    # Call local_multi_tensor_scale (copy + scale)
    local_multi_tensor_scale(None, None, [src_list, dst_list], scale)

    # Verify correctness
    for t, expected in zip(dst_list, expected_list):
        torch.testing.assert_close(t, expected)

def test_local_multi_tensor_scale_empty():
    scale = 0.5
    src_list = []
    dst_list = []

    # Should not raise error
    local_multi_tensor_scale(None, None, [src_list, dst_list], scale)

if __name__ == "__main__":
    test_local_multi_tensor_scale_inplace()
    test_local_multi_tensor_scale_copy()
    test_local_multi_tensor_scale_copy_scale_1()
    test_local_multi_tensor_scale_empty()
