# Month 0, Week 1: High-Performance Python Internals & Systems Engineering — Exercises

```
================================================================================
COURSE:    AI & LLM SYSTEMS ENGINEERING (2026 EDITION)
MODULE:    MONTH 0: ADVANCED PYTHON & APPLIED MATHEMATICS FOR AI
WEEK:      WEEK 1: HIGH-PERFORMANCE PYTHON INTERNALS & UV TOOLING
DELIVERABLE: RIGOROUS GRADUATE-LEVEL LAB EXERCISES & VERIFICATION SUITE
TARGET:    SENIOR AI ENGINEERS & SYSTEMS ARCHITECTS
================================================================================
```

---

## Overview & Execution Protocol

These exercises are designed to test and solidify your mechanistic understanding of CPython runtime internals, memory representations, garbage collection dynamics, bytecode execution, and custom iterator protocols.

### Environment Setup
All exercises must be implemented using **Python 3.11+** managed strictly via **uv**:
```bash
uv init week1-exercises
cd week1-exercises
uv add pytest numpy tabulate
```

### Graded Difficulty Levels
- **Tier 1: Core Mechanics (Questions 1.1 – 1.3)**: Memory addresses, small integer caching, ref counting, and mutability.
- **Tier 2: Intermediate Systems (Questions 1.4 – 1.6)**: Bytecode disassembly, circular reference garbage collection, and custom memory profiling.
- **Tier 3: Advanced Engineering (Questions 1.7 – 1.8)**: High-performance zero-copy iterators and custom memory pool allocators.

---

## Tier 1: Core Mechanics & CPython Memory Internals

### Exercise 1.1: Small Integer Caching & String Interning Probe
- **Objective**: Mechanistically demonstrate CPython's integer pre-allocation pool ($[-5, 256]$) and string interning mechanisms using `id()` and the `is` identity operator.
- **Requirements**:
  1. Write a function `probe_small_integer_cache(start: int = -10, end: int = 300) -> tuple[int, int]` that returns the exact lowest and highest integer boundaries where `a is b` evaluates to `True` for independently instantiated integers.
  2. Explain why `a = 256; b = 256; a is b` is `True`, but `a = 257; b = 257; a is b` is `False` in interactive REPLs, yet may evaluate to `True` within a compiled code block.
  3. Use `sys.intern()` to demonstrate how runtime string identity comparisons can be optimized from $O(N)$ string character scans to $O(1)$ pointer comparisons.

---

### Exercise 1.2: Reference Counting & Object Graph Inspection
- **Objective**: Track the physical CPython reference count (`PyObject.ob_refcnt`) of dynamic objects across scope transitions.
- **Requirements**:
  1. Write a script utilizing `sys.getrefcount()` to track an object's reference counter through:
     - Variable assignment.
     - Insertion into a list, dictionary, and tuple.
     - Passing as an argument to a function.
     - Deletion via `del`.
  2. Explain why `sys.getrefcount(x)` always returns a count that is **1 higher** than the actual number of references holding `x`.
  3. Construct a test demonstrating how shallow copies (`copy.copy()`) increment reference counters of nested elements, whereas deep copies (`copy.deepcopy()`) allocate new heap objects.

---

### Exercise 1.3: The Mutable Default Argument Catastrophe
- **Objective**: Deconstruct why default function arguments are evaluated at **function definition time** rather than **invocation time**.
- **Requirements**:
  1. Write a function with a signature `def append_tensor(tensor: list, cache: list = []) -> list`. Demonstrate how successive calls mutate the shared default list across completely unrelated invocations.
  2. Inspect the function's `__defaults__` tuple in Python bytecode before and after invocation.
  3. Refactor the function to the production-standard sentinel pattern (`cache: list | None = None`) and prove that new heap instances are allocated per invocation.

---

## Tier 2: Intermediate Systems & CPython Bytecode

### Exercise 1.4: Bytecode Disassembly & Stack Machine Analysis
- **Objective**: Analyze the execution stack of CPython's virtual machine using the standard `dis` module.
- **Requirements**:
  1. Disassemble the following two functions that achieve the same result:
     ```python
     def loop_sum(n: int) -> int:
         s = 0
         for i in range(n):
             s += i
         return s

     def comp_sum(n: int) -> int:
         return sum(i for i in range(n))
     ```
  2. Identify the specific bytecode opcodes (`LOAD_FAST`, `STORE_FAST`, `BINARY_OP`, `FOR_ITER`).
  3. Explain which function executes fewer bytecode instructions and why list/generator comprehensions execute in isolated C-stack frames.

---

### Exercise 1.5: Circular References & Generational Garbage Collection Tuning
- **Objective**: Force a circular reference cycle that bypasses reference counting and requires the generational garbage collector (`gc`).
- **Requirements**:
  1. Define a class `Node` that holds a pointer to another `Node`. Create a cycle: `a.next = b; b.next = a`.
  2. Explicitly disable automatic GC using `gc.disable()`.
  3. Delete references `del a, b`. Prove that `sys.getrefcount()` cannot reclaim these objects, causing a silent memory leak.
  4. Call `gc.collect()` and verify that the circular reference is successfully identified and collected in Generation 2. Inspect `gc.garbage`.

---

### Exercise 1.6: Memory Profiling via `tracemalloc`
- **Objective**: Measure peak heap memory allocation and identify line-by-line memory allocation bottlenecks in Python.
- **Requirements**:
  1. Write a function that processes 100,000 dictionary records in two ways:
     - Approach A: Creating a list of full Python dictionaries `[{"id": i, "val": i * 2} for i in range(100_000)]`.
     - Approach B: Using `__slots__` or a namedtuple / dataclass.
  2. Use `tracemalloc.start()` and `tracemalloc.take_snapshot()` to profile both approaches.
  3. Output a tabular comparison showing: Total RAM allocated (MB), Peak RAM (MB), and relative memory savings ($>60\%$).

---

## Tier 3: Advanced Systems Engineering

### Exercise 1.7: High-Performance Zero-Copy Chunking Iterator
- **Objective**: Implement a memory-efficient sliding-window iterator over large sequences without copying data.
- **Requirements**:
  1. Implement a class `ZeroCopySlidingWindow(sequence, window_size: int, step: int)` implementing the `__iter__` and `__next__` dunder protocols.
  2. The iterator must utilize Python `memoryview` or `itertools.islice` to ensure that window slices **never allocate new duplicate strings or byte arrays in memory**.
  3. Benchmark processing a 50MB byte buffer: compare the peak memory footprint of `ZeroCopySlidingWindow` against naive list slicing `seq[i:i+window_size]`.

---

### Exercise 1.8: Custom Fixed-Size Object Memory Pool
- **Objective**: Implement an object pooler in Python to eliminate memory allocator overhead (`PyObject_Malloc`) during high-frequency short-lived object lifecycles.
- **Requirements**:
  1. Construct a class `TensorPayloadPool(object_factory, pool_size: int = 1000)`.
  2. Implement `acquire()` and `release(obj)` methods using a thread-safe deque.
  3. Benchmark 1,000,000 allocation-destruction cycles: compare standard object instantiation time against pool recycling time.

---

## Solutions & Verification Key

### Solution 1.1: Small Integer Cache Probe
```python
import sys

def probe_small_integer_cache(start: int = -10, end: int = 300) -> tuple[int, int]:
    cached_ints = []
    for i in range(start, end):
        # Create two separate instances
        a = int(str(i))
        b = int(str(i))
        if a is b:
            cached_ints.append(i)
    return min(cached_ints), max(cached_ints)

if __name__ == "__main__":
    low, high = probe_small_integer_cache()
    print(f"CPython Integer Cache Range: [{low}, {high}]")
    assert low == -5 and high == 256, f"Expected [-5, 256], got [{low}, {high}]"
```

### Solution 1.5: Circular Reference GC Verification
```python
import gc
import sys

class CycleNode:
    def __init__(self, name: str):
        self.name = name
        self.next = None

def verify_cycle_gc():
    gc.collect()
    gc.disable() # Disable cyclic collector

    a = CycleNode("A")
    b = CycleNode("B")
    a.next = b
    b.next = a

    del a, b # Reference count drops from 2 to 1 (not zero!)
    
    # Prove objects still reside in memory
    uncollected = gc.collect() # Manually trigger collector
    print(f"Cyclic objects collected by gc: {uncollected}")
    assert uncollected >= 2, "Failed to collect circular references!"
    gc.enable()

if __name__ == "__main__":
    verify_cycle_gc()
```

