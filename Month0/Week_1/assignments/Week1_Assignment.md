# Month 0, Week 1: High-Performance Python Internals & Systems Engineering — Assignment

```
================================================================================
COURSE:    AI & LLM SYSTEMS ENGINEERING (2026 EDITION)
MODULE:    MONTH 0: ADVANCED PYTHON & APPLIED MATHEMATICS FOR AI
WEEK:      WEEK 1: HIGH-PERFORMANCE PYTHON INTERNALS & UV TOOLING
PROJECT:   ENTERPRISE IN-MEMORY TELEMETRY BUFFER & CACHE PROFILER
DELIVERABLE: PRODUCTION PYTHON PACKAGE, BENCHMARK REPORT & TEST SUITE
TARGET:    SENIOR AI ENGINEERS & SYSTEMS ARCHITECTS
================================================================================
```

---

## 1. Project Overview & Architectural Mission

In modern AI and LLM inference engines (such as vLLM, TensorRT-LLM, or distributed data loaders), naive Python object allocations create severe performance degradation. High-frequency object allocations trigger frequent Python garbage collection stop-the-world pauses, fragment heap memory, and choke CPU cache lines.

For this assignment, you will architect an **Enterprise In-Memory Telemetry Buffer & Object Pooler** (`telemetry_buffer`). This system must safely ingest, store, window, and serialize millions of streaming high-frequency AI inference telemetry events while enforcing:
1. **Zero-Copy Memory Utilization**: Leveraging `memoryview`, byte buffers, and `__slots__` to achieve $\ge 60\%$ memory reduction over standard dictionary representations.
2. **Deterministic Memory Pooling**: Eliminating `PyObject_Malloc` overhead by recycling fixed-size telemetry payload structures.
3. **CPython Reference Safety**: Demonstrating zero circular reference leaks and validating complete garbage collection cleanup.
4. **Professional Modern Tooling**: Managed exclusively with **uv**, formatted with **ruff**, type-checked with **mypy (strict mode)**, and verified via **pytest** with $\ge 85\%$ coverage.

---

## 2. Technical Specification & Architecture

```
+---------------------------------------------------------------------------------------------------+
|                        TELEMETRY BUFFER ARCHITECTURAL WORKFLOW                                    |
+---------------------------------------------------------------------------------------------------+
|  High-Frequency Stream: (Inference Events: model_id, prompt_tokens, latency_ms, status)           |
|       |                                                                                           |
|       v                                                                                           |
|  [Object Pooler: TelemetryPayloadPool]                                                            |
|       - Pre-allocates N reusable __slots__ instances                                              |
|       - Eliminates runtime dynamic heap allocations                                               |
|       |                                                                                           |
|       v                                                                                           |
|  [Circular Ring Buffer: ZeroCopyRingBuffer]                                                       |
|       - Fixed-capacity circular array avoiding list reallocation resizing                         |
|       - O(1) insertion, O(1) FIFO eviction, O(1) sliding window slicing                           |
|       |                                                                                           |
|       v                                                                                           |
|  [Memory Profiler & Verification Suite]                                                           |
|       - tracemalloc peak memory verification vs dict baselines                                    |
|       - Pytest automated verification suite (>=85% test coverage)                                 |
+---------------------------------------------------------------------------------------------------+
```

### Module Structure
Your submission must follow this exact repository structure:
```
telemetry_buffer/
├── pyproject.toml               # uv package configuration with ruff, mypy, pytest
├── README.md                    # Benchmark report and architectural rationale
├── src/
│   └── telemetry/
│       ├── __init__.py
│       ├── payload.py           # Slots-based memory-optimized telemetry records
│       ├── pool.py              # Thread-safe object recycler pool
│       ├── buffer.py            # Circular ring buffer with sliding-window iterators
│       └── profiler.py          # tracemalloc comparison harness
└── tests/
    ├── __init__.py
    ├── test_payload.py          # Memory footprint and slots verification
    ├── test_pool.py             # Acquisition, recycling, and capacity bounds
    └── test_buffer.py           # O(1) insertion, eviction, and zero-copy slicing
```

---

## 3. Implementation Requirements

### Requirement 1: Memory-Optimized Telemetry Payload (`src/telemetry/payload.py`)
- Implement a class `TelemetryRecord`:
  - Attributes: `event_id` (str), `timestamp` (float), `prompt_tokens` (int), `completion_tokens` (int), `latency_ms` (float), `status_code` (int).
  - Must define `__slots__ = (...)` to eliminate the default per-instance `__dict__` overhead.
  - Implement a method `to_bytes() -> bytes` using Python `struct.pack()` to serialize records into a compact binary format.
  - Implement a `classmethod` `from_bytes(data: bytes) -> TelemetryRecord` using `struct.unpack()`.

### Requirement 2: Deterministic Object Pooler (`src/telemetry/pool.py`)
- Implement a class `TelemetryPayloadPool`:
  - `__init__(self, capacity: int = 10_000)`: Pre-allocates instances into an internal deque.
  - `acquire(self, **kwargs) -> TelemetryRecord`: Retrieves an idle record, populates its attributes, and returns it. If the pool is empty, dynamically instantiates a new record.
  - `release(self, record: TelemetryRecord) -> None`: Resets the record's attributes to defaults and returns it to the free pool.
  - Must be thread-safe using `threading.Lock`.

### Requirement 3: High-Performance Circular Ring Buffer (`src/telemetry/buffer.py`)
- Implement a class `ZeroCopyRingBuffer`:
  - `__init__(self, max_capacity: int = 100_000)`: Pre-allocates a fixed-size list `[None] * max_capacity` to prevent dynamic memory resizing.
  - `append(self, record: TelemetryRecord) -> None`: Inserts the record in $O(1)$ time. If the buffer is full, overwrites the oldest element (circular FIFO).
  - `get_window(self, size: int) -> list[TelemetryRecord]`: Returns the most recent `size` records without copying the entire buffer array.
  - Implement the `__iter__` protocol to yield records in chronological order from oldest to newest.

### Requirement 4: Benchmarking & Memory Profiling (`src/telemetry/profiler.py`)
- Write an automated benchmark comparing:
  - **Baseline**: 100,000 standard Python dictionaries stored in a naive list.
  - **Optimized**: 100,000 `TelemetryRecord` instances managed via `ZeroCopyRingBuffer`.
- Output must record:
  - Peak RAM footprint in Megabytes (using `tracemalloc`).
  - Total allocation time in milliseconds.
  - Serialization speed comparison (JSON serialization vs. binary `struct.pack`).

---

## 4. Verification & Grading Rubric

Your assignment will be evaluated across 5 strict engineering dimensions:

| Dimension | Weight | Acceptance Threshold |
| :--- | :---: | :--- |
| **1. Memory Optimization** | 25% | `TelemetryRecord` utilizes `__slots__`; achieves $\ge 60\%$ RAM reduction over standard dictionaries in `tracemalloc` profiling. |
| **2. Algorithmic Rigor** | 25% | Ring buffer executes insertions and evictions in strictly $O(1)$ time; object pool recycles instances without memory leaks. |
| **3. Type Safety & Linting** | 20% | `mypy --strict` passes with 0 errors; code formatted with `ruff` conforming to PEP 8 standards. |
| **4. Test Suite Coverage** | 20% | Comprehensive `pytest` test suite achieving $\ge 85\%$ branch and statement coverage. |
| **5. Technical Report** | 10% | `README.md` includes empirical benchmark tables, CPython memory analysis, and architectural trade-off discussion. |

---

## 5. Starter Code Template

```python
# src/telemetry/payload.py
import struct
from typing import ClassVar

class TelemetryRecord:
    __slots__ = (
        "event_id",
        "timestamp",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "status_code"
    )

    # Struct format: 16s (UUID), d (double ts), i (int prompt), i (int comp), f (float latency), h (short status)
    STRUCT_FORMAT: ClassVar[str] = "!16sdiifh"

    def __init__(
        self,
        event_id: str = "",
        timestamp: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        status_code: int = 200
    ) -> None:
        self.event_id = event_id
        self.timestamp = timestamp
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms
        self.status_code = status_code

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.event_id.encode("utf-8")[:16].ljust(16, b"\x00"),
            self.timestamp,
            self.prompt_tokens,
            self.completion_tokens,
            self.latency_ms,
            self.status_code
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TelemetryRecord:
        eid_bytes, ts, p_tok, c_tok, lat, status = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(
            event_id=eid_bytes.decode("utf-8").rstrip("\x00"),
            timestamp=ts,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            latency_ms=lat,
            status_code=status
        )
```

---

## 6. Submission Guidelines
1. Push your solution to your assigned repository branch: `submission/month0-week1`.
2. Include the output of `pytest --cov=src` in your PR description.
3. Attach the benchmark results table generated by `python -m telemetry.profiler`.
