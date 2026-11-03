# Month 0, Week 1 Cheatsheet: CPython Runtime Internals, Memory Layout & Data Structures

---

## 1. CPython Object Model Architecture (`PyObject`)

In CPython, every variable is a pointer to a heap-allocated `PyObject`.

```
                  Standard PyObject (64-bit Architecture)
 0                   16                  24                  32 bytes
┌───────────────────┬───────────────────┬───────────────────┐
│     ob_refcnt     │     ob_type       │    Extra fields   │
│     (8 bytes)     │    (8 bytes)      │    (Payload...)   │
│ uint64 reference  │ pointer to type   │                   │
└───────────────────┴───────────────────┴───────────────────┘

                  Variable-Length PyVarObject (e.g. tuple, list)
 0                   16                  24                  32 bytes
┌───────────────────┬───────────────────┬───────────────────┬───────────────┐
│     ob_refcnt     │     ob_type       │      ob_size      │ Array items.. │
│     (8 bytes)     │    (8 bytes)      │ uint64 item count │ Pointers (8B) │
└───────────────────┴───────────────────┴───────────────────┴───────────────┘
```

### Critical Memory Sizing Rules
```python
import sys

sys.getsizeof(object())  # 16 bytes: base PyObject header (8B ob_refcnt + 8B ob_type)
sys.getsizeof(0)         # 28 bytes: PyLongObject (16B header + 8B ob_size + 4B digit)
sys.getsizeof("")        # 49-80 bytes: PyASCIIObject header + null terminator
sys.getsizeof([])        # 56 bytes: empty list (header + capacity counter + pointer buffer)
sys.getsizeof({})        # 64 bytes: compact hash table header (indices + entries)
```

---

## 2. Integer Caching, Interning & Memory Pointers

### Small Integer Cache (`[-5, 256]`)
CPython pre-allocates an array of 312 singletons during runtime startup.
```python
a = 256
b = 256
a is b          # True (same physical address in small integer array)

x = 257
y = 257
x is y          # False in interactive REPL (distinct heap allocations)
id(x) == id(y)  # False
```

### String Interning Mechanics
- String literals matching `[a-zA-Z0-9_]*` are automatically interned into `interned_dict`.
- Dynamically created strings are not interned unless explicitly requested:
```python
import sys
s1 = sys.intern("custom_token_" + str(42))
s2 = sys.intern("custom_token_42")
s1 is s2        # True (instant O(1) pointer comparison)
```

---

## 3. Data Structure Memory & Algorithmic Complexities

| Data Structure | Underlying C Layout | Amortized Access | Amortized Insert | Memory Overhead |
|:---|:---|:---|:---|:---|
| **`list`** | Dynamic `PyObject**` array | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ append / $\mathcal{O}(n)$ insert | Over-allocates by $\sim 12.5\%$ |
| **`tuple`** | Fixed `PyObject*[]` buffer | $\mathcal{O}(1)$ | Immutable | Zero excess buffer |
| **`deque`** | Doubly-linked block nodes (64 items) | $\mathcal{O}(n)$ random | $\mathcal{O}(1)$ head/tail push | Low memory fragmentation |
| **`dict`** | Split-table / Compact hash table | $\mathcal{O}(1)$ avg | $\mathcal{O}(1)$ amortized | $2/3$ max load factor |
| **`set`** | Flat open-addressing table | $\mathcal{O}(1)$ avg | $\mathcal{O}(1)$ amortized | Hash + key array |

### List Over-Allocation Formula in CPython
When resizing, CPython grows capacity via:
$$\text{new\_allocated} = \text{new\_size} + (\text{new\_size} \gg 3) + (\text{new\_size} < 9 \mathrel{?} 3 : 6)$$

---

## 4. `__slots__` Optimization vs Default `__dict__`

### Default Class Instance (`__dict__`)
```python
class DefaultNode:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Memory per instance: ~152 bytes (instance header + dict header + values array)
```

### Slotted Class Instance (`__slots__`)
Eliminates dynamic `__dict__`, storing fields directly at static struct offsets.
```python
class SlottedNode:
    __slots__ = ("x", "y")
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Memory per instance: ~48 bytes (68% RAM savings for 10M+ objects)
```

---

## 5. Garbage Collection & Reference Counting Mechanics

```
               Object Lifecycle Decision Tree
                         ┌──────────────┐
                         │ Object Born  │
                         └──────┬───────┘
                                │
                        ob_refcnt == 0 ?
                       ┌────────┴────────┐
                       ▼                 ▼
                     [YES]              [NO]
               Deallocate via        Contained in
               PyObject_Free()     Cyclic Reference?
                                     ┌───┴───┐
                                     ▼       ▼
                                   [YES]    [NO]
                               Tracked by  Remains in
                                GC Gen 0   Memory
```

### Cyclic Garbage Collector Tri-Color Generational Bounds
- **Generation 0**: Allocations exceeding `threshold_0` triggers scan.
- **Generation 1**: Surviving Gen 0 objects promoted.
- **Generation 2**: Long-lived objects; full cycle collection.

```python
import gc

gc.get_threshold()          # (700, 10, 10)
gc.collect()                # Force full generational collection
gc.disable()                # Optimize batch ETL pipelines with no circular refs
```

---

## 6. Common Pitfalls & Debugging Playbook

### Trap 1: Default Mutable Arguments
```python
# BUG: List is created once at function definition time in func.__defaults__
def append_event(event, event_log=[]):
    event_log.append(event)
    return event_log

# FIX: Use None sentinel pattern
def append_event(event, event_log=None):
    if event_log is None:
        event_log = []
    event_log.append(event)
    return event_log
```

### Trap 2: In-Place Modification vs Rebidding
```python
a = [1, 2, 3]
b = a
a += [4]        # In-place extends a; b is also modified!
a = a + [5]     # Allocates a new list; a and b diverge!
```

---

## 7. High-Performance Memory Profiling Quick Commands

```python
import tracemalloc

tracemalloc.start()
# Execute candidate ML memory workload...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

for stat in top_stats[:5]:
    print(f"{stat.traceback}: {stat.size / 1024:.2f} KB (allocated {stat.count} objects)")
```
