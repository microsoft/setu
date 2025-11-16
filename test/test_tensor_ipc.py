"""
Unit tests for Tensor IPC (Inter-Process Communication) functionality.

Tests the ability to share CUDA tensors between processes using IPC handles.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor


# Module-level function for multiprocessing (must be picklable)
def _child_process_modify_tensor(spec_dict, result_queue):
    """Child process that rebuilds and modifies the tensor."""
    try:
        # Rebuild tensor from spec
        args = {
            **spec_dict,
            'tensor_cls': torch.Tensor,
            'storage_cls': torch.storage.UntypedStorage
        }
        rebuilt_tensor = rebuild_cuda_tensor(**args)

        # Read original value
        original_value = rebuilt_tensor[0][0][0].item()

        # Modify the tensor
        rebuilt_tensor[0][0][0] = 42.0

        # Verify modification
        modified_value = rebuilt_tensor[0][0][0].item()

        # Send results back to parent
        result_queue.put({
            'success': True,
            'original_value': original_value,
            'modified_value': modified_value
        })
    except Exception as e:
        import traceback
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@pytest.mark.gpu
def test_tensor_ipc_basic():
    """Test basic tensor IPC sharing between processes."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from setu._commons.utils import prepare_tensor_ipc_spec

    # Create a shared tensor
    original_tensor = torch.randn((1, 2, 3), device='cuda')
    original_value = original_tensor[0][0][0].item()

    # Prepare IPC spec
    spec = prepare_tensor_ipc_spec(original_tensor)
    spec_dict = spec.to_dict()

    # Verify spec contains expected fields
    assert 'tensor_size' in spec_dict
    assert 'tensor_stride' in spec_dict
    assert 'tensor_offset' in spec_dict
    assert 'dtype' in spec_dict
    assert 'requires_grad' in spec_dict
    assert 'storage_device' in spec_dict
    assert 'storage_handle' in spec_dict
    assert 'storage_size_bytes' in spec_dict
    assert 'storage_offset_bytes' in spec_dict
    assert 'ref_counter_handle' in spec_dict
    assert 'ref_counter_offset' in spec_dict
    assert 'event_handle' in spec_dict
    assert 'event_sync_required' in spec_dict

    # Verify tensor properties
    assert tuple(spec_dict['tensor_size']) == (1, 2, 3)
    assert spec_dict['storage_device'] == 0  # Should be device index (int)
    assert isinstance(spec_dict['storage_handle'], bytes)
    assert isinstance(spec_dict['ref_counter_handle'], bytes)
    assert isinstance(spec_dict['event_handle'], bytes)

@pytest.mark.gpu
def test_tensor_ipc_cross_process_modification():
    """Test that modifications in one process are visible in another."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from setu._commons.utils import prepare_tensor_ipc_spec

    # Create a tensor in parent process
    parent_tensor = torch.zeros((1, 2, 3), device='cuda')
    initial_value = parent_tensor[0][0][0].item()

    # Prepare IPC spec
    spec = prepare_tensor_ipc_spec(parent_tensor)
    spec_dict = spec.to_dict()

    # Create queue for results
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    # Spawn child process
    process = ctx.Process(
        target=_child_process_modify_tensor,
        args=(spec_dict, result_queue)
    )
    process.start()
    process.join(timeout=10)

    # Check if process completed successfully
    if process.exitcode != 0:
        pytest.fail(f"Child process failed with exit code {process.exitcode}")

    # Get results from child process
    result = None
    try:
        result = result_queue.get(timeout=5)
    except Exception as e:
        pytest.fail(f"Failed to get result from child process: {e}")

    assert result is not None, "No result received from child process"

    if not result['success']:
        error_msg = result.get('error', 'Unknown')
        traceback_msg = result.get('traceback', '')
        pytest.fail(f"Child process error: {error_msg}\n{traceback_msg}")

    # Verify child process saw the original value
    assert abs(result['original_value'] - initial_value) < 1e-6, \
        f"Child saw {result['original_value']}, expected {initial_value}"

    # Verify child process modified the value
    assert abs(result['modified_value'] - 42.0) < 1e-6, \
        f"Child modified to {result['modified_value']}, expected 42.0"

    # CRITICAL: Verify parent process can see the modification
    parent_current_value = parent_tensor[0][0][0].item()
    assert abs(parent_current_value - 42.0) < 1e-6, \
        f"Parent cannot see modification: expected 42.0, got {parent_current_value}"

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--gpu'])
