# Setu: Global Tensor Registry (GTR)

## The Problem

Modern large-scale ML systems need to move tensors efficiently across heterogeneous components. This is more complex than simple byte-array transfers, because:

* Structure: Tensors have dimensions, cardinalities, and shared layouts, and moving them requires respecting these semantics.
* Parallelism mismatches: Different components may shard tensors differently. Today, the application is responsible for manually handling reshaping and resharding, potentially breaking abstraction layers.
* Topology and transport complexities: Efficiency in movement depends on hardware topology, which could force the application to manage communication strategies, further complicating the simple goal of “moving a tensor.”
* Development complexity: The logic for resharding, layout transformation, communication planning, and topology handling is reinvented and reimplemented, and scatters the performance-critical decisions across multiple components. Abstracting these concerns into a separate component (i.e., the GTR) frees the application to focus on semantics of movement, and avoid repeated (and sometimes inconsistent) efforts.

## The Solution

We propose a “Global Tensor Registry” (GTR) that abstracts tensor copies over arbitrary *parallelism schemes* and *hardware topologies*. A parallelism scheme defines how a tensor is sharded or replicated across workers. Applications describe only the semantic move, while the system resolves how to realize it efficiently in practice.

When an application requests a copy (e.g., “move slice X of tensor A to tensor B”), the GTR uses a centralized metadata-store (“Metastore”), to interpret the source and destination layouts, then compiles a logical-to-physical plan. The plan encodes how to reshard, consolidate, and distribute the tensor across workers while exploiting topology-aware communication backends (e.g., NCCL, etc.).

## Example: KV Cache Transfer

Consider a system implementing PD-disaggregation, where prefill workers run with TP=2 and decode nodes running with TP=4. Both sides use the same KV cache tensor (semantically; although the paging layout might be different), but with different shard layouts. With the GTR, the application can simply state the intent, i.e., “move pages \[`p0`, .., `pn`\] of `prefill:0/layer:0/kv` to `decode:1/layer:0/kv`,” and the system resolves the layouts, perform resharding, and selects the optimal transport strategy automatically.

## Core Contributions

Our core contributions will be:

1. Capturing parallelism scheme in a declarative, structured format.
2. Enable parallelism scheme agnostic tensor store engine.
3. Leverage domain specific knowledge captured in the parallelism scheme to enable optimization passes that result in performant physical copy plans.

## Design

### Goals & Non-goals for V1

**Goals:**

* High-level interface for tensors (fancy indexing, named dimensions[^1]).
* Source-initiated copy that enforces ownership at source.
* Dense tensors with contiguous shards.
* Schema is immutable. Data is versioned per tensor.

**Non-Goals:**

* Replicated or non-contiguous ownership (e.g., paged, block-cyclic, etc.).
* Compression, encryption.

[^1]: PyTorch has support for named tensors, as described [here](https://docs.pytorch.org/docs/stable/named_tensor.html). We can leverage this feature for better usability.


### Terminology

- **Device:** A physical placement target for tensor memory (e.g., `cuda:0`).
- **Node:** A physical machine with one or more devices.


### Core System Components

1. **Client SDK:**
    - Entrypoint for applications.
    - Allows clients to register tensor shards and issue high-level copy commands (`CopyIntent`).

2. **MetaStore:**
    - Centralized catalog of all tensors, dimensions, ownership mappings, and worker layouts.

3. **Planner:**
    - Compiles high-level copy intents into an executable plan.
    - Uses metadata, layouts, and topology information.
    - Optimizes the plan with strategies like packing, chunking, pipelining, and routing.

4. **Node Agent:**
    - Per-node control process.
    - Registers local tensors.
    - Accepts `CopyIntent` requests from clients.
    - Executes plan fragments by coordinating with other NodeAgents.

5. **Execution Runtimes:**
    - A collection of low-level transfer engines (e.g., NIXL, NCCL, etc.).
    - Used by Node Agents to perform actual data movement.


### Data Model

```cpp
using Index = int64_t;
using Size = uint64_t;

using IndexSet = std::set<Index>;
struct Range { Index start; Index end; }; // [start, end)

using TensorName = std::string; // ex.: "prefill:0/layer:0/kv"
using DimensionName = std::string; // ex.: "page", "seq", "head"
using TensorDimensions = std::unordered_map<DimensionName, Size>; // unsharded tensor dimensions
using OwnedTensorDimensions = std::vector<std::pair<DimensionName, Range>>;
```

### Sharded Tensor Layout

```cpp
enum class DType { FP16, BF16, FP32, I8, ... }

struct Device {
    enum class Kind { CPU, CUDA, ... }

    Kind kind;
    int device_index;
}

struct DevicePtr {
    Device device;
    uintptr_t address; // base address
}

/*
 * Layout describes the physical layout of a tensor shard on a device.
 */
struct Layout {
    DType dtype;
    DevicePtr base_ptr;

    std::vector<DimensionName> dims;
    std::vector<Size> shape;
    std::vector<Index> strides;

    // logical to physical ordering
    std::unordered_map<DimensionName, size_t> logical_to_physical_dim;

    uint64_t element_offset_local(const std::vector<Index>& idx) {
        // assert(0 <= idx[i] < shape[i]) for all i;

        uint64_t offset = 0;
        for (size_t i = 0; i < idx.size(); ++i) offset += idx[i] * strides[i];
        return offset;
    }
}
```

### Data Versioning

#### Model

- Schema (& ownership) is immutable.
- Single writer per tensor.
- Data versions are monotonically increasing integers.

### Slicing Model

```cpp
using DimSelect = std::variant<Range, IndexSet>;
using SliceSpec = std::unordered_map<DimensionName, DimSelect>;
```

> [!NOTE]
> Omitted dimension names in `SliceSpec` imply selecting the entire dimension.

### Copy Intent

A *copy intent* declares *what* to copy.

```cpp
struct TensorRef {
    TensorName name;
    std::optional<Version> version; // optional versioning
}

struct CopyIntent {
    TensorRef src;
    SliceSpec src_slice;

    TensorRef dst;
    SliceSpec dst_slice;

    struct Options {
        std::optional<std::chrono::milliseconds> timeout;
        std::optional<uint8_t> priority; // higher value -> higher priority
        // ... other options
    }
}
```

> [!NOTE]
> - `dtype(src) == dtype(dst)`.
>   - In the future, we can consider support for type casting.
> - Element counts of `src_slice` and `dst_slice` must match.
> - All selected source elements must be owned by the source worker.

### MetaStore

The MetaStore is the single source of truth for tensor schemas and shard ownerships.

```cpp
struct ShardDescriptor {
    TensorName tensor_name;
    OwnedTensorDimensions owned_dims; // owned ranges for each dimension
    Layout layout;                    // physical layout on device
}

using DataVersion = uint64_t;
using NodeAgentID = std::string; // globally unique ID for a NodeAgent
using ShardRecord = std::pair<NodeAgentID, ShardDescriptor>;

struct TensorMetadata {
    TensorName name;
    TensorDimensions dims;
    DType dtype;
    DataVersion data_version;

    std::vector<ShardRecord> shards;
}
```

> [!NOTE]
> The node agents register shards, the unsharded tensor is never registered explicitly.
> When all shards of a tensor are registered, the tensor is considered fully registered.

```cpp
enum class ErrorCode {
}

struct Error {
    ErrorCode code;
    std::string message;
}

service MetaStore {
    // Registration agents and keeping them alive
    std::expected<void, Error> RegisterNodeAgent(NodeAgentID id, std::string addr);
    std::expected<void, Error> Heartbeat(NodeAgentID id);

    // Schema & ownership
    std::expected<void, Error> RegisterShard(
        TensorName name,
        TensorDimensions dims,
        OwnedTensorDimensions owned_dims,
        Layout layout
    );

    std::expected<TensorMetadata, Error> GetTensorMetadata(TensorName name);

    // Some sketches for how versioning might work.
    // The following API is incomplete.
    std::expected<DataVersion, Error> BeginPublish(TensorName name);
    std::expected<void, Error> EndPublish(TensorName name, DataVersion version);
}
```

### Planner

Purpose: compile a `CopyIntent` into a per-worker program that realize the move with correct ownership, data version, and optimal use of the topology.

```cpp
struct TopologyInfo {
    using LinkType = std::string; // ex.: "infiniband", "ethernet", "nvlink", ...

    std::vector<NodeAgentID> workers;
    std::unordered_map<LinkType, std::vector<std::vector<double>>> pairwise_latency_us;
    std::unordered_map<LinkType, std::vector<std::vector<double>>> pairwise_bandwidth_gbps;

    // link_types[i][j] = set of link types between workers[i] and workers[j]
    std::vector<std::vector<std::vector<LinkType>>> link_types;
}
```




`some possible node types: Pack, Unpack, Alloc, Free, Write (as in RDMAWrite), Send

#### Possible optimizations:

1. `Packing/unpacking: Whether to pack on source worker and/or unpack on destination worker`
2. `Chunking and pipelining`
3. `Routing to intermediate workers for hierarchical topologies`

#### Options:

1. `Each worker’s NodeAgent receives a program from centralized planner to execute the plan`
2. ~~`Each source worker’s NodeAgent receives a program to execute the plan and each destination worker’s NodeAgent receives instructions (from source workers) to execute`~~
3. `Each worker’s NodeAgent receives instructions from centralized planner to execute`
4. `All of the above`

#### PlanChunk:

```cpp
struct PlanChunk {
    WorkerID src
    WorkerID dst
    TensorName src
    TensorName dst

    // Slices must be contiguous
    SliceSpec src_slice
    SliceSpec dst_slice

    ...
}

struct ExecutionPlan {
    ...
}

service Planner {
    expected<ExecutionPlan, Error>
    MakePlan(CopyIntent intent, // User specified
        List[OwnershipMap] src_own_maps, // Rest from the system
        List[OwnershipMap] dst_own_maps,
        List[Layout] src_layouts,
        List[Layout] dst_layouts,
        TopologyInfo topology)
}

struct TopologyInfo { // specific to set of src and dest workers`
    Map<LinkType, Matrix2d<int>> pairwise_latency;
    Map<LinkType, Matrix2d<int>> pairwise_bandwidth;
    Matrix2d<Set<LinkType>> link_type;
}
```

### NodeAgent

The 'transfer agent' runs on each worker and is responsible for coordinating with other workers to execute data movement plans.

```cpp
service NodeAgent {
    expected<void, Error>
    Initialize(...)

    expected<void, Error>
    RegisterLocal(TensorMetadata meta, OwnershipMap my_slice)

    future<expected<void, Error>>
    Copy(CopyIntent intent)
}
```

### Execution Runtime

The execution runtime is the physical transfer engine. It provides the low-level mechanisms for moving tensor “fragments” (i.e., a continuous chunk of a tensor) once the Planner has emitted an execution DAG.

### Client SDK

The client connects to the local `GTRAgent`.

// XXX: Safe vs. Unsafe API

```python
import gts
import torch

# Client creation
client = gts.Client("10.0.0.1:6174")

# Registering a tensor
t = client.create_and_register(
    name="replica:0/worker:0/task:0/t",
    dims=[("a", 2048, (0, 1024)), ("b", 128, (0, 128)), ("c", 256, (0, 256)),
    # (name, size, owned_range(s))
    dtype=torch.bfloat16,
    device="cuda:0"
)


#----------------------------------------
# PREFILL
#----------------------------------------
kv_caches[/*layer_id*/=0] = client.create_and_register(
    name="prefill_replica:0/layer:0/kv_cache",
    dims=[("a", 2048, (0, 1024)),
          ("b", 128, (0, 64)),
          ("c", 256, (0, 256))],
    # (name, size, owned_range(s))
    dtype=torch.bfloat16,
    device=f"cuda:{gpu_idx}"
)

#----------------------------------------
# DECODE
#----------------------------------------
kv_caches[/*layer_id*/=0] = client.create_and_register(
    name="decode_replica:0/layer:0/kv_cache",
    dims=[("a", 2048, (0, 1024)),
          ("b", 128, (0, 64)),
          ("c", 256, (0, 256))],
    # (name, size, owned_range(s))
    dtype=torch.bfloat16,
    device="cuda:0"
)

#----------------------------------------
# TRANSFERING (ON PREFIL)
#----------------------------------------

# XXX:
src = client.tensor("prefill_replica:0/layer:0/kv_cache") \
            .where(a=src_page_ids)                        \
            .where(b=gts.ALL)                             \
            .where(c=gts.slice(0, 32))                    \
            .build()

# OR
src = kv_cache[/*layer_id=*/0].where(a=src_page_ids)...

dst = client.tensor("decode_replica:0/layer:0/kv_cache") \
            .where(a=dst_page_ids)                       \
            .where(b=gts.ALL)                            \
            .where(c=gts.slice(0, 32))                   \
            .build()

future = client.copy(src, dst, options={})
future.wait()

#----------------------------------------

# Semantic copying

src = client.tensor("worker:0/task:0/t") \
            .where(a=src_page_ids)     \
            .where(b=gts.ALL)          \
            .where(c=gts.slice(0, 32)) \
            .build()

dst = client.tensor("worker:1/task:1/t")

future = client.copy(src, dst, options={})
future.wait()
```

## Optimization passes

### Attempt 1:

```python
# input: list of copy intents
# output: list of copy intents
def transform(copy_intents: List[CopyIntent]) → List[CopyIntent]: ...

# a transformation pass is a sequence of operations on copy intents
# a transformation pass knows the underlying topology
# goal? let's say to reduce e2e batch time
# for simplicity, assume each copy intent in the list is executed sequentially

# Operations on CopyIntent

# Fuse two copy intents
def fuse(ca: CopyIntent, cb: CopyIntent) → CopyIntent:
    assert ca.src == cb.src and ca.dst == cb.dst
    assert is_contiguous(ca.src_slice.index, cb.src_slice.index)
    assert is_contiguous(ca.dst_slice.index, cb.dst_slice.index)
    return CopyIntent(
        src=ca.src,
        src_slice=concat(ca.src_slice, cb.src_slice),
        dst=ca.dst,
        dst_slice=concat(ca.dst_slice, cb.dst_slice),
    )

# Split a copy intent into n copy intents
def split(c: CopyIntent, n: int) → List[CopyIntent]: …

# Example scenario:
# [CopyIntent(a[0:10], b[0:10]), CopyIntent(c[0:5], d[0:5]), CopyIntent(a[10:15], b[10:15])]
# fuse 0 2
# [CopyIntent(a[0:15], b[0:15]), CopyIntent(c[0:5], d[0:5])]
```

okay, keeping the optimization domain restricted to CopyIntent limits us to mainly fuse operations… why would you ever split?

splitting would make sense if you could run a group of copy intents in parallel. Certain groupings are more efficient than others.

let's introduce a GroupedCopyIntent… a collection of CopyIntents that can run together.

### Attempt 2:

Attempt 1 does not factor in exploiting parallelism schemes of the intents. right now, we are only considering sharding.

## Previous version:

We wish to develop a tensor store that enables tensor copies parameterized over *parallelism schemes* and *topologies*. A parallelization scheme defines how a tensor is *sharded* and *replicated*.

Tensor stores today only allow a pre-defined set of tensor-to-tensor copies between parallelization schemes. The engine has intricate knowledge of how the copy is to be orchestrated at a physical level. When a new parallelism scheme is discovered, the tensor store must be changed to support it.

Our idea is to abstract out the parallelism scheme, passing it as a parameter to the tensor store. We then compile a logical to physical copy plan, using knowledge of the parallelism scheme as a guide.

### Example:

- `tensor_1`: Parallelism scheme - Equally shard into 4, replicate each shard 2 times

- `tensor_2`: Parallelism scheme - Replicate 2 times

Then, `copy(tensor_1, tensor_2)` compiles to:

```python
for tensor_1_shard in tensor_1:
    tensor_1_shard_x = tensor_1_shard[0] # suffices to read from one replica
    for tensor_2_replica in tensor_2:
        physical_copy(tensor_1_shard_x, tensor_2_replica[tensor_1_shard_x.start:tensor_1_shard_x.end])
```

The above compilation is just an example. There could be more optimized ways of orchestrating the code.

Our core contributions will be:

1. Capturing parallelism scheme in a declarative, structured format
2. Enable parallelism scheme agnostic tensor store engine
3. Leverage domain specific knowledge captured in the parallelism scheme to enable optimization passes that result in performant physical copy plans

## Parallelism Schemes

```python
type parallelism_scheme =
  | Split of int * parallelism_scheme
  | Replicate of int * parallelism_scheme
  | Base
```

```python
let fmap f tensor parallelism_scheme =
  letrec helper t ps =
    match ps with
    | Base → fmap f t
    | Split (ps', n) →
    let shards = shard t n in
    let t' = foldl (fun acc x →
      concat acc (f x)
    ) shards empty_tensor in
    helper t' ps'
    | Replicate (ps, n) →
    helper t ps
  in
    helper tensor parallelism_scheme

```python
let sample_scheme =
  Split (4, Base)
  |> Replicate 2
  |> Split 3
```

```python
let sample_tensor =
    Runtime.create_tensor sample_scheme
```

```python
let () =
    fmap (fun x → x + 1) sample_tensor sample_scheme
```

### VLLM KV Connector

[vllm kv connector](https://docs.google.com/document/d/1WFxgkY0Zr1NmTdWn7numry2M-fqod8AY7NfHjVmLVo0/edit?usp=sharing)