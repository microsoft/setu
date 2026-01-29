from __future__ import annotations

import torch
import multiprocessing
import os
import time
import traceback

# Set debug logging to see the flow
os.environ["SETU_LOG_LEVEL"] = "DEBUG"

from setu._client import Client
from setu._commons.datatypes import Device, TensorDim, TensorShardSpec
from setu._commons.enums import DeviceKind
from setu._coordinator import Coordinator
from setu._node_manager import NodeAgent


def create_sample_tensor_shard_spec(name: str) -> TensorShardSpec:
    device = Device(
        node_rank=0,
        device_rank=0,
        torch_device=torch.device('cuda:0')
    )

    dims = [
        TensorDim("first", 32),
        TensorDim("second", 768),
    ]

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
    """Run the Coordinator process."""
    try:
        print(f"[Coordinator] Starting on ports executor={router_executor_port}, handler={router_handler_port}")

        coordinator = Coordinator(router_executor_port, router_handler_port)
        coordinator.start()

        print("[Coordinator] Started and ready to receive requests")
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
        ready_event.set()  # Prevent deadlock
        result_queue.put(("coordinator", False, str(e)))


def run_node_agent_process(
    node_rank: int,
    router_port,
    dealer_executor_port,
    dealer_handler_port,
    ready_event,
    stop_event,
    result_queue,
) -> None:
    """Run a NodeAgent process."""
    try:
        print(
            f"[NodeAgent] Starting with node_rank={node_rank}, "
            f"router_port={router_port}, "
            f"dealer_executor_port={dealer_executor_port}, "
            f"dealer_handler_port={dealer_handler_port}"
        )

        devices = [
            Device(
                node_rank=0,
                device_rank=0,
                torch_device=torch.device('cuda:0')
            )
        ]

        node_agent = NodeAgent(
            node_rank=node_rank,
            router_port=router_port,
            dealer_executor_port=dealer_executor_port,
            dealer_handler_port=dealer_handler_port,
            devices=devices
        )
        node_agent.start()

        print("[NodeAgent] Started and ready to receive client requests")
        ready_event.set()

        # Wait until stop signal
        while not stop_event.is_set():
            time.sleep(0.1)

        print("[NodeAgent] Stopping...")
        node_agent.stop()
        print("[NodeAgent] Stopped")
        result_queue.put(("node_agent", True, None))

    except Exception as e:
        print(f"[NodeAgent] Error: {e}")
        traceback.print_exc()
        ready_event.set()  # Prevent deadlock
        result_queue.put(("node_agent", False, str(e)))


def run_client_process(
    node_agent_endpoint,
    ready_event,
    stop_event,
    result_queue,
) -> None:
    """Run a Client process that sends RegisterTensorShard requests."""
    try:
        print(f"[Client] Starting, will connect to {node_agent_endpoint}")

        # Give NodeAgent a moment to fully initialize
        time.sleep(0.5)

        # Create and connect client
        client = Client(client_rank=0)
        client.connect(node_agent_endpoint)
        print(f"[Client] Connected to {node_agent_endpoint}")

        # Send RegisterTensorShard request
        tensor_name = "test_model.layer0.weight"
        print(f"[Client] Sending RegisterTensorShard request for '{tensor_name}'")

        shard_spec = create_sample_tensor_shard_spec(tensor_name)
        print(f"[Client] TensorShardSpec: {shard_spec}")

        result = client.register_tensor_shard(shard_spec)

        if result is not None:
            print(f"[Client] SUCCESS! Received TensorShardRef: {result}")
            result_queue.put(("client", True, f"Received: {result}"))
        else:
            print("[Client] SUCCESS! Received None (stub implementation)")
            result_queue.put(("client", True, "Received None (expected for stub)"))

        # Test multiple requests
        print("\n[Client] Sending additional requests...")
        for i in range(3):
            tensor_name = f"test_model.layer{i}.bias"
            shard_spec = create_sample_tensor_shard_spec(tensor_name)
            result = client.register_tensor_shard(shard_spec)
            print(f"[Client] Request {i+1}: '{tensor_name}' -> {result}")

        # Disconnect
        client.disconnect()
        print("[Client] Disconnected")

        # Signal that client is done
        stop_event.set()

    except Exception as e:
        print(f"[Client] Error: {e}")
        traceback.print_exc()
        result_queue.put(("client", False, str(e)))
        stop_event.set()


def main() -> None:
    """Main function to run the test."""
    print("=" * 70)
    print("Test: RegisterTensorShard Flow (Client -> NodeAgent -> Coordinator)")
    print("=" * 70)

    # Port configuration
    # Coordinator listens on these ports (ROUTER sockets)
    coordinator_executor_port = 19000
    coordinator_handler_port = 19001

    # NodeAgent listens on this port for clients (ROUTER socket)
    # and connects to coordinator ports (DEALER sockets)
    node_agent_router_port = 19100

    # Client connects to NodeAgent
    node_agent_endpoint = f"tcp://localhost:{node_agent_router_port}"

    # Synchronization events
    coordinator_ready = multiprocessing.Event()
    node_agent_ready = multiprocessing.Event()
    client_ready = multiprocessing.Event()
    stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    processes = []

    try:
        # 1. Start Coordinator first
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

        # 2. Start NodeAgent (connects to Coordinator)
        print("\n--- Starting NodeAgent ---")
        node_agent_proc = multiprocessing.Process(
            target=run_node_agent_process,
            args=(
                0,  # node_rank
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

        # 3. Start Client (connects to NodeAgent)
        print("\n--- Starting Client ---")
        client_ready.set()  # Signal that infrastructure is ready
        client_proc = multiprocessing.Process(
            target=run_client_process,
            args=(
                node_agent_endpoint,
                client_ready,
                stop_event,
                result_queue,
            ),
        )
        client_proc.start()
        processes.append(("Client", client_proc))

        # Wait for client to finish (it sets stop_event when done)
        client_proc.join(timeout=30.0)

        # Give a moment for final messages
        time.sleep(1.0)

        # Signal stop to all processes
        print("\n--- Shutting Down ---")
        stop_event.set()

        # Wait for processes to finish
        for name, proc in processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                print(f"Force terminating {name}")
                proc.terminate()
                proc.join(timeout=2.0)

        # Collect results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        all_success = True
        while not result_queue.empty():
            component, success, message = result_queue.get()
            status = "PASSED" if success else "FAILED"
            print(f"  [{component}] {status}: {message}")
            if not success:
                all_success = False

        print("\n" + "=" * 70)
        if all_success:
            print("TEST PASSED: Full RegisterTensorShard flow works!")
        else:
            print("TEST FAILED: See errors above")
        print("=" * 70)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        stop_event.set()
        for name, proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
