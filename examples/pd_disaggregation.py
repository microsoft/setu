"""
Example: Prompt/Decode (PD) Disaggregation using Setu Client API

This example demonstrates how to use Setu's tensor selection and copy APIs
to implement disaggregated prompt and decode phases in LLM inference.

Architecture:
- Prefill replica: Processes prompts and generates initial KV cache
- Decode replica: Processes token generation using transferred KV cache

The key operation is copying KV cache pages from prefill to decode replicas.
"""

import torch

from setu._commons.datatypes import TensorDim
from setu.client import Client


def setup_kv_cache_shards(client: Client, num_layers: int = 32) -> None:
    """
    Create KV cache tensor shards for prefill and decode replicas.

    Args:
        client: Setu client instance
        num_layers: Number of transformer layers
    """
    # KV cache dimensions
    # - page: Paged attention block index (2048 pages total)
    # - head: Attention head dimension (32 heads)
    # - seq: Sequence length per page (16 tokens/page)
    # - hidden: Hidden dimension per head (128)

    for layer_id in range(num_layers):
        # Prefill replica KV cache
        prefill_shard = client.create_tensor_shard(
            name=f"prefill_replica:0/layer:{layer_id}/kv_cache",
            dims=[
                TensorDim("page", 2048),  # Total pages in system
                TensorDim("head", 32),  # Multi-head attention
                TensorDim("seq", 16),  # Tokens per page
                TensorDim("hidden", 128),  # Hidden dim per head
            ],
            dtype=torch.bfloat16,
            device="cuda:0",
        )

        # Decode replica KV cache (same structure)
        decode_shard = client.create_tensor_shard(
            name=f"decode_replica:0/layer:{layer_id}/kv_cache",
            dims=[
                TensorDim("page", 2048),
                TensorDim("head", 32),
                TensorDim("seq", 16),
                TensorDim("hidden", 128),
            ],
            dtype=torch.bfloat16,
            device="cuda:1",
        )

    print(f"Created KV cache shards for {num_layers} layers")


def transfer_kv_cache(
    client: Client, src_page_ids: list[int], dst_page_ids: list[int], layer_id: int
) -> None:
    """
    Transfer KV cache pages from prefill to decode replica.

    This is the core operation in PD disaggregation - moving computed KV cache
    from the prefill replica (which processed the prompt) to the decode replica
    (which will generate tokens).

    Args:
        client: Setu client instance
        src_page_ids: Page IDs in prefill replica to copy from
        dst_page_ids: Page IDs in decode replica to copy to
        layer_id: Transformer layer index
    """
    # Build source selection: specific pages, all heads/seq/hidden implicitly selected
    # Using list indexing (numpy-like)
    src_selection = (
        client.select(f"prefill_replica:0/layer:{layer_id}/kv_cache")
        .where("page", src_page_ids)  # List indexing
    )

    # Build destination selection: corresponding pages in decode replica
    dst_selection = (
        client.select(f"decode_replica:0/layer:{layer_id}/kv_cache")
        .where("page", dst_page_ids)  # List indexing
    )

    # Execute the copy operation (internally handles device-to-device transfer)
    client.copy(src_selection, dst_selection)

    print(
        f"Layer {layer_id}: Transferred {len(src_page_ids)} pages "
        f"from prefill to decode replica"
    )


def partial_kv_update(
    client: Client, page_id: int, layer_id: int, new_tokens: torch.Tensor
) -> None:
    """
    Demonstrate partial KV cache update - updating specific sequence positions.

    This shows fine-grained tensor selection for in-place updates.

    Args:
        client: Setu client instance
        page_id: Page to update
        layer_id: Layer index
        new_tokens: New KV values for specific sequence positions [num_tokens, head, hidden]
    """
    num_tokens = new_tokens.shape[0]

    # Select specific page and sequence range to update
    # Note: heads and hidden are implicitly selected (all indices)
    update_selection = (
        client.select(f"decode_replica:0/layer:{layer_id}/kv_cache")
        .where("page", page_id)  # Single index
        .where("seq", slice(0, num_tokens))  # Slice indexing (numpy-like)
    )

    # TODO: Once write handles are implemented, update would be:
    # with client.write(shard) as tensor:
    #     # Apply selection and write new_tokens
    #     tensor[update_selection] = new_tokens

    print(
        f"Layer {layer_id}, Page {page_id}: "
        f"Updated first {num_tokens} sequence positions"
    )


def main() -> None:
    """Main example demonstrating PD disaggregation workflow."""
    # Initialize client
    client = Client()

    # Setup: Create KV cache shards for all layers
    NUM_LAYERS = 32
    setup_kv_cache_shards(client, num_layers=NUM_LAYERS)

    print("\n--- Prefill Phase ---")
    # Simulate prefill: process prompt and fill KV cache pages
    # In reality, these would be computed by the prefill replica
    prefill_pages = [0, 1, 2, 3, 4]  # Pages used for prompt processing
    print(f"Prefill processed prompt using pages: {prefill_pages}")

    print("\n--- Transfer Phase ---")
    # Transfer KV cache from prefill to decode replica
    # This is the key operation in PD disaggregation
    decode_pages = [100, 101, 102, 103, 104]  # Destination pages in decode replica

    for layer_id in range(NUM_LAYERS):
        transfer_kv_cache(client, prefill_pages, decode_pages, layer_id)

    print("\n--- Decode Phase ---")
    # Simulate decode: generate tokens using transferred KV cache
    print(f"Decode replica generating tokens using pages: {decode_pages}")

    # Example: Update KV cache with newly generated token
    new_kv_data = torch.randn(1, 32, 128, dtype=torch.bfloat16)  # 1 token, 32 heads
    partial_kv_update(client, page_id=decode_pages[0], layer_id=0, new_tokens=new_kv_data)

    print("\n--- Advanced: Selective Transfer ---")
    # More complex scenario: Transfer only specific attention heads
    # Note: seq and hidden are implicitly selected (all indices)
    selective_src = (
        client.select("prefill_replica:0/layer:0/kv_cache")
        .where("page", [5, 6, 7])  # List indexing
        .where("head", [0, 1, 2, 3])  # Only first 4 heads
    )

    selective_dst = (
        client.select("decode_replica:0/layer:0/kv_cache")
        .where("page", [105, 106, 107])  # List indexing
        .where("head", [0, 1, 2, 3])
    )

    client.copy(selective_src, selective_dst)
    print("Transferred 3 pages x 4 heads (selective copy)")


if __name__ == "__main__":
    main()