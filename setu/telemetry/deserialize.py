"""Binary deserializer matching C++ BinaryWriter format.

Setu serialization conventions:
- Little-endian for all integers and floats.
- Strings: uint32 length prefix + raw bytes (UTF-8).
- Vectors: uint32 count prefix + elements.
- UUIDs: 16 raw bytes (boost::uuids::uuid layout).
- Variants: uint32 index prefix + active alternative value.
- Serializable objects: uint32 byte-size prefix + serialized payload.
"""

import struct
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


class BinaryReader:
    """Forward-only binary reader matching C++ BinaryWriter format."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def _read_raw(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._data):
            raise ValueError(
                f"BinaryReader overflow: need {n} bytes at offset {self._pos}, "
                f"have {len(self._data) - self._pos}"
            )
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def read_uint32(self) -> int:
        return struct.unpack("<I", self._read_raw(4))[0]

    def read_uint64(self) -> int:
        return struct.unpack("<Q", self._read_raw(8))[0]

    def read_int32(self) -> int:
        return struct.unpack("<i", self._read_raw(4))[0]

    def read_double(self) -> float:
        return struct.unpack("<d", self._read_raw(8))[0]

    def read_string(self) -> str:
        length = self.read_uint32()
        return self._read_raw(length).decode("utf-8")

    def read_int64(self) -> int:
        return struct.unpack("<q", self._read_raw(8))[0]

    def read_uint64_native(self) -> int:
        """Read a std::size_t (8 bytes on 64-bit systems)."""
        return struct.unpack("<Q", self._read_raw(8))[0]

    def read_bool(self) -> bool:
        return struct.unpack("<?", self._read_raw(1))[0]

    def read_uuid(self) -> uuid.UUID:
        raw = self._read_raw(16)
        return uuid.UUID(bytes=raw)

    def remaining(self) -> int:
        return len(self._data) - self._pos


# ---------------------------------------------------------------------------
# Deserialized metric types (mirrors C++ structs)
# ---------------------------------------------------------------------------


@dataclass
class PassTiming:
    pass_name: str
    elapsed_ms: float


@dataclass
class NCCLGroupTiming:
    group_index: int
    elapsed_ms: float
    num_ops: int


@dataclass
class NCCLWorkerMetricsRecord:
    copy_op_id: uuid.UUID
    node_id: uuid.UUID
    device_rank: int
    group_timings: List[NCCLGroupTiming]
    total_execute_ms: float


@dataclass
class CompilationMetricsRecord:
    copy_op_id: uuid.UUID
    total_compile_time_ms: float
    pass_timings: List[PassTiming]
    num_participants: int
    participant_instruction_counts: List[Tuple[str, int]]


@dataclass
class IndexRangeRecord:
    start: int
    end: int


@dataclass
class IndexRangeSetRecord:
    dim_size: int
    ranges: List[IndexRangeRecord]


@dataclass
class TensorSelectionRecord:
    name: str
    indices: Dict[str, IndexRangeSetRecord]


@dataclass
class E2EMetricsRecord:
    copy_op_id: uuid.UUID
    e2e_time_ms: float
    total_bytes_transferred: int
    src_name: str
    dst_name: str
    src_selection: Optional[TensorSelectionRecord]
    dst_selection: Optional[TensorSelectionRecord]


# Variant index mapping (must match C++ MetricsMessage variant order)
_VARIANT_INDEX_NCCL_WORKER = 0
_VARIANT_INDEX_COMPILATION = 1
_VARIANT_INDEX_E2E = 2


def _read_pass_timing(reader: BinaryReader) -> PassTiming:
    """Read a PassTiming (Serializable: uint32 size prefix + payload)."""
    _size = reader.read_uint32()
    name = reader.read_string()
    ms = reader.read_double()
    return PassTiming(pass_name=name, elapsed_ms=ms)


def _read_nccl_group_timing(reader: BinaryReader) -> NCCLGroupTiming:
    """Read a NCCLGroupTiming (Serializable: uint32 size prefix + payload)."""
    _size = reader.read_uint32()
    idx = reader.read_uint32()
    ms = reader.read_double()
    ops = reader.read_uint64()
    return NCCLGroupTiming(group_index=idx, elapsed_ms=ms, num_ops=ops)


def _read_nccl_worker_metrics(reader: BinaryReader) -> NCCLWorkerMetricsRecord:
    """Read NCCLWorkerMetrics (Serializable: uint32 size prefix + payload)."""
    _size = reader.read_uint32()
    copy_op_id = reader.read_uuid()
    node_id = reader.read_uuid()
    device_rank = reader.read_int32()

    num_timings = reader.read_uint32()
    timings = [_read_nccl_group_timing(reader) for _ in range(num_timings)]

    total_ms = reader.read_double()
    return NCCLWorkerMetricsRecord(
        copy_op_id=copy_op_id,
        node_id=node_id,
        device_rank=device_rank,
        group_timings=timings,
        total_execute_ms=total_ms,
    )


def _read_compilation_metrics(reader: BinaryReader) -> CompilationMetricsRecord:
    """Read CompilationMetrics (Serializable: uint32 size prefix + payload)."""
    _size = reader.read_uint32()
    copy_op_id = reader.read_uuid()
    total_ms = reader.read_double()

    num_timings = reader.read_uint32()
    timings = [_read_pass_timing(reader) for _ in range(num_timings)]

    num_participants = reader.read_uint32()

    num_entries = reader.read_uint32()
    counts = []
    for _ in range(num_entries):
        name = reader.read_string()
        count = reader.read_uint32()
        counts.append((name, count))

    return CompilationMetricsRecord(
        copy_op_id=copy_op_id,
        total_compile_time_ms=total_ms,
        pass_timings=timings,
        num_participants=num_participants,
        participant_instruction_counts=counts,
    )


def _read_index_range_set(reader: BinaryReader) -> IndexRangeSetRecord:
    """Read an IndexRangeSet (Serializable: uint32 size prefix + payload).

    Payload: size_t dim_size + vector<IndexRange>.
    IndexRange is trivially copyable (int64 start + int64 end = 16 bytes),
    so the vector is bulk-copied: uint32 count + count * 16 bytes.
    """
    _size = reader.read_uint32()
    dim_size = reader.read_uint64_native()
    num_ranges = reader.read_uint32()
    ranges = []
    for _ in range(num_ranges):
        start = reader.read_int64()
        end = reader.read_int64()
        ranges.append(IndexRangeRecord(start=start, end=end))
    return IndexRangeSetRecord(dim_size=dim_size, ranges=ranges)


def _read_tensor_selection(reader: BinaryReader) -> Optional[TensorSelectionRecord]:
    """Read a shared_ptr<TensorSelection>.

    Wire format: bool present + (if true) Serializable TensorSelection.
    TensorSelection payload: string name + unordered_map<string, IndexRangeSet>.
    """
    present = reader.read_bool()
    if not present:
        return None
    # Serializable size prefix
    _size = reader.read_uint32()
    name = reader.read_string()
    num_dims = reader.read_uint32()
    indices: Dict[str, IndexRangeSetRecord] = {}
    for _ in range(num_dims):
        dim_name = reader.read_string()
        irs = _read_index_range_set(reader)
        indices[dim_name] = irs
    return TensorSelectionRecord(name=name, indices=indices)


def _read_e2e_metrics(reader: BinaryReader) -> E2EMetricsRecord:
    """Read E2EMetrics (Serializable: uint32 size prefix + payload)."""
    _size = reader.read_uint32()
    copy_op_id = reader.read_uuid()
    ms = reader.read_double()
    total_bytes = reader.read_uint64()
    src_name = reader.read_string()
    dst_name = reader.read_string()
    src_selection = _read_tensor_selection(reader)
    dst_selection = _read_tensor_selection(reader)
    return E2EMetricsRecord(
        copy_op_id=copy_op_id,
        e2e_time_ms=ms,
        total_bytes_transferred=total_bytes,
        src_name=src_name,
        dst_name=dst_name,
        src_selection=src_selection,
        dst_selection=dst_selection,
    )


# Maps variant index to reader function
_VARIANT_READERS = {
    _VARIANT_INDEX_NCCL_WORKER: _read_nccl_worker_metrics,
    _VARIANT_INDEX_COMPILATION: _read_compilation_metrics,
    _VARIANT_INDEX_E2E: _read_e2e_metrics,
}


def deserialize_metrics_message(data: bytes):
    """Deserialize a MetricsMessage from raw bytes.

    Returns one of: NCCLWorkerMetricsRecord, CompilationMetricsRecord,
    E2EMetricsRecord.
    """
    reader = BinaryReader(data)
    variant_index = reader.read_uint32()

    reader_fn = _VARIANT_READERS.get(variant_index)
    if reader_fn is None:
        raise ValueError(f"Unknown MetricsMessage variant index: {variant_index}")

    return reader_fn(reader)
