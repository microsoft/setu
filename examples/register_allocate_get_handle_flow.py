import os
import torch
import multiprocessing
import time
import traceback
from torch.multiprocessing.reductions import rebuild_cuda_tensor

os.environ["SETU_LOG_LEVEL"] = "DEBUG"

from setu._client import Client
from setu._commons.datatypes import Device, TensorDim, TensorShardSpec
from setu._coordinator import Coordinator
from setu._node_manager import NodeAgent


def create_tensor_shard_spec(name, shape) -> TensorShardSpec:
    """Create a TensorShardSpec for testing."""
    device = Device(
        node_rank=0,
        device_rank=0,
        torch_device=torch.device("cuda:0"),
    )

    dims = [TensorDim(f"dim_{i}", size) for i, size in enumerate(shape)]

    return TensorShardSpec(
        name=name,
        dims=dims,
        dtype=torch.float32,
        device=device,
    )


def run_coordinator_process(
    router_executor_port,
    router_handler_port,
    ready_event,
    stop_event,
    result_queue,
) -> None:
    try:
        print(f"[Coordinator] Starting on ports executor={router_executor_port}, handler={router_handler_port}")
        coordinator = Coordinator(router_executor_port, router_handler_port)
        coordinator.start()
        print("[Coordinator] Started and ready")
        ready_event.set()

        while not stop_event.is_set():
            time.sleep(0.1)

        print("[Coordinator] Stopping...")
        coordinator.stop()
        print("[Coordinator] Stopped")
        result_queue.put(("coordinator", True, None))

    except Exception as e:
        print(f"[Coordinator] Error: {e}")
        traceback.print_exc()
        ready_event.set()
        result_queue.put(("coordinator", False, str(e)))


def run_node_agent_process(
    node_rank,
    router_port,
    dealer_executor_port,
    dealer_handler_port,
    ready_event,
    stop_event,
    result_queue,
) -> None:
    try:
        print(f"[NodeAgent] Starting with router_port={router_port}")
        devices = [
            Device(
                node_rank=0,
                device_rank=0,
                torch_device=torch.device("cuda:0"),
            )
        ]

        node_agent = NodeAgent(
            node_rank=node_rank,
            router_port=router_port,
            dealer_executor_port=dealer_executor_port,
            dealer_handler_port=dealer_handler_port,
            devices=devices,
        )
        node_agent.start()
        print("[NodeAgent] Started and ready")
        ready_event.set()

        while not stop_event.is_set():
            time.sleep(0.1)

        print("[NodeAgent] Stopping...")
        node_agent.stop()
        print("[NodeAgent] Stopped")
        result_queue.put(("node_agent", True, None))

    except Exception as e:
        print(f"[NodeAgent] Error: {e}")
        traceback.print_exc()
        ready_event.set()
        result_queue.put(("node_agent", False, str(e)))


def run_client_register_get_handle_and_modify(
    client_id,
    node_agent_endpoint,
    tensor_name,
    tensor_shape,
    modification_value,
    ready_event,
    result_queue,
) -> None:
    try:
        print(f"[Client {client_id}] Waiting for infrastructure to be ready...")
        ready_event.wait(timeout=10)
        time.sleep(0.5)

        print(f"[Client {client_id}] Connecting to {node_agent_endpoint}")
        client = Client(client_rank=client_id)
        client.connect(node_agent_endpoint)
        print(f"[Client {client_id}] Connected")

        print(f"[Client {client_id}] Registering tensor '{tensor_name}' with shape {tensor_shape}")
        shard_spec = create_tensor_shard_spec(tensor_name, tensor_shape)
        shard_ref = client.register_tensor_shard(shard_spec)

        if shard_ref is None:
            print(f"[Client {client_id}] Failed to register tensor (got None)")
            result_queue.put((f"client_{client_id}", False, "Failed to register tensor"))
            client.disconnect()
            return

        print(f"[Client {client_id}] Registered tensor, shard_ref: {shard_ref}")

        print(f"[Client {client_id}] Getting tensor handle...")
        tensor_ipc_spec = client.get_tensor_handle(tensor_name)
        spec_dict = tensor_ipc_spec.to_dict()
        print(f"[Client {client_id}] Got tensor handle with size {spec_dict['tensor_size']}")

        # Rebuild the tensor from IPC handle and modify it
        print(f"[Client {client_id}] Rebuilding tensor from IPC handle...")
        args = {
            **spec_dict,
            "tensor_cls": torch.Tensor,
            "storage_cls": torch.storage.UntypedStorage,
        }
        rebuilt_tensor = rebuild_cuda_tensor(**args)

        original_value = rebuilt_tensor.flatten()[0].item()
        print(f"[Client {client_id}] Original tensor value: {original_value}")

        print(f"[Client {client_id}] Filling tensor with {modification_value}...")
        rebuilt_tensor.fill_(modification_value)

        modified_value = rebuilt_tensor.flatten()[0].item()
        print(f"[Client {client_id}] Modified tensor value: {modified_value}")

        client.disconnect()
        print(f"[Client {client_id}] Disconnected")
        result_queue.put((f"client_{client_id}", True, {
            "spec_dict": spec_dict,
            "original_value": original_value,
            "modified_value": modified_value,
            "expected_value": modification_value,
        }))

    except Exception as e:
        print(f"[Client {client_id}] Error: {e}")
        traceback.print_exc()
        result_queue.put((f"client_{client_id}", False, str(e)))


def run_client_verify_tensor(
    client_id: int,
    spec_dict: dict,
    expected_value: float,
    result_queue,
) -> None:
    try:
        print(f"[Verifier {client_id}] Rebuilding tensor from IPC handle...")
        args = {
            **spec_dict,
            "tensor_cls": torch.Tensor,
            "storage_cls": torch.storage.UntypedStorage,
        }
        rebuilt_tensor = rebuild_cuda_tensor(**args)

        actual_value = rebuilt_tensor.flatten()[0].item()
        print(f"[Verifier {client_id}] Actual value: {actual_value}, Expected: {expected_value}")

        success = abs(actual_value - expected_value) < 1e-5
        result_queue.put(
            (
                f"verifier_{client_id}",
                success,
                {
                    "actual_value": actual_value,
                    "expected_value": expected_value,
                },
            )
        )

    except Exception as e:
        print(f"[Verifier {client_id}] Error: {e}")
        traceback.print_exc()
        result_queue.put((f"verifier_{client_id}", False, str(e)))


def main():
    """Main function to run the e2e tensor IPC example."""
    print("=" * 70)
    print("End-to-End Tensor IPC Example")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    # Port configuration
    coordinator_executor_port = 19500
    coordinator_handler_port = 19501
    node_agent_router_port = 19502
    node_agent_endpoint = f"tcp://localhost:{node_agent_router_port}"

    # Synchronization events
    coordinator_ready = multiprocessing.Event()
    node_agent_ready = multiprocessing.Event()
    stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    processes = []

    try:
        # 1. Start Coordinator
        print("\n--- Starting Coordinator ---")
        coordinator_proc = multiprocessing.Process(
            target=run_coordinator_process,
            args=(
                coordinator_executor_port,
                coordinator_handler_port,
                coordinator_ready,
                stop_event,
                result_queue,
            ),
        )
        coordinator_proc.start()
        processes.append(("Coordinator", coordinator_proc))

        if not coordinator_ready.wait(timeout=10):
            print("ERROR: Coordinator failed to start")
            return

        # 2. Start NodeAgent
        print("\n--- Starting NodeAgent ---")
        node_agent_proc = multiprocessing.Process(
            target=run_node_agent_process,
            args=(
                0,
                node_agent_router_port,
                coordinator_executor_port,
                coordinator_handler_port,
                node_agent_ready,
                stop_event,
                result_queue,
            ),
        )
        node_agent_proc.start()
        processes.append(("NodeAgent", node_agent_proc))

        if not node_agent_ready.wait(timeout=10):
            print("ERROR: NodeAgent failed to start")
            return

        # 3. Client registers tensor, gets handle, and modifies it
        print("\n--- Client: Register Tensor, Get Handle, and Modify ---")
        modification_value = 123.0
        client_result_queue = multiprocessing.Queue()
        client_proc = multiprocessing.Process(
            target=run_client_register_get_handle_and_modify,
            args=(
                0,
                node_agent_endpoint,
                "test_tensor",
                (4, 8),
                modification_value,
                node_agent_ready,
                client_result_queue,
            ),
        )
        client_proc.start()
        client_proc.join(timeout=30)

        client_name, client_success, client_result = client_result_queue.get(timeout=5)
        if not client_success:
            print(f"ERROR: Client failed: {client_result}")
            return

        spec_dict = client_result["spec_dict"]
        print(f"SUCCESS: Got tensor IPC spec with size {spec_dict['tensor_size']}")
        print(f"SUCCESS: Tensor modified from {client_result['original_value']} to {client_result['modified_value']}")

        # 4. Verifier process verifies the modification
        print("\n--- Verifier: Verify Tensor Value ---")
        verifier_result_queue = multiprocessing.Queue()
        verifier_proc = multiprocessing.Process(
            target=run_client_verify_tensor,
            args=(0, spec_dict, modification_value, verifier_result_queue),
        )
        verifier_proc.start()
        verifier_proc.join(timeout=10)

        ver_name, ver_success, ver_result = verifier_result_queue.get(timeout=5)
        if not ver_success:
            print(f"ERROR: Verifier failed: expected {ver_result['expected_value']}, got {ver_result['actual_value']}")
            return

        print(f"SUCCESS: Verified tensor value is {ver_result['actual_value']}")

        # Signal stop
        print("\n--- Shutting Down ---")
        stop_event.set()

        # Wait for processes to finish
        time.sleep(1.0)
        for name, proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                print(f"Force terminating {name}")
                proc.terminate()
                proc.join(timeout=2)

        print("\n" + "=" * 70)
        print("SUCCESS: End-to-End Tensor IPC Example Completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        stop_event.set()
        for name, proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
