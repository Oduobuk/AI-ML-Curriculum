# Month 0, Week 2 Cheatsheet: Advanced Control Flow, Decorators, Asyncio & Metaprogramming

---

## 1. Closure Mechanics & Lexical Scope

A closure bundles a code object with an environment of bound references via cell objects:
```python
def make_counter(start: int = 0):
    count = start
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

c = make_counter(10)
c()                        # 11
c.__closure__[0].cell_contents  # 11 (inspects enclosed variable in heap cell)
```

### The Late-Binding Trap in Loop Factories
```python
# FAULTY: All functions bind to the final mutated value of i (i = 3)
funcs = [lambda x: x * i for i in range(4)]
[f(2) for f in funcs]      # [6, 6, 6, 6]

# FIXED 1: Default argument capture (evaluated at definition time)
funcs_fixed = [lambda x, i=i: x * i for i in range(4)]
[f(2) for f in funcs_fixed]  # [0, 2, 4, 6]

# FIXED 2: Immediate invocation factory
funcs_factory = [(lambda val: lambda x: x * val)(i) for i in range(4)]
```

---

## 2. Production Decorators & Signature Preservation

Always use `@functools.wraps` to preserve `__name__`, `__doc__`, `__annotations__`, and `__wrapped__`.

```python
import functools
import inspect
import time
from typing import Callable, Any, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def robust_retry(max_attempts: int = 3, delay_sec: float = 0.1):
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(fn)
        
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Validate parameters against signature
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise RuntimeError(f"{fn.__name__} exhausted {max_attempts} retries") from exc
                    time.sleep(delay_sec * attempt)
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator
```

---

## 3. Bidirectional Generator Protocol (`send`, `throw`, `close`)

Generators act as state machines yielding values and receiving inputs:
```python
def streaming_accumulator():
    total = 0.0
    while True:
        # yield current total, pause, and receive incoming scalar
        val = yield total
        if val is None:
            break
        total += val

acc = streaming_accumulator()
next(acc)        # 0.0 (Prime the generator)
acc.send(10.0)   # 10.0
acc.send(25.5)   # 35.5
acc.close()      # Terminates generator loop
```

---

## 4. Asyncio Event Loop & Cooperative Concurrency

```
                      Asyncio Event Loop Lifecycle
     ┌────────────────────────────────────────────────────────┐
     │                Ready Queue (Runnable Tasks)            │
     └──────────────────────────┬─────────────────────────────┘
                                │
                          Picks next task
                                │
                                ▼
                       ┌─────────────────┐
                       │ Task Execution  │
                       └────────┬────────┘
                                │
                        await expression?
                       ┌────────┴────────┐
                       ▼                 ▼
                     [YES]              [NO]
               Suspends & registers   Task runs to
               IO/Timer Selector      StopIteration
```

### Essential Asyncio Coordination Patterns
```python
import asyncio

# 1. Bounded Concurrency
sem = asyncio.Semaphore(10)
async def worker():
    async with sem:
        await fetch_upstream()

# 2. Structured TaskGroup (Python 3.11+)
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(query_db())
    t2 = tg.create_task(query_cache())
# Automatically awaits both; raises ExceptionGroup on failure

# 3. Offloading Blocking CPU Code
res = await asyncio.to_thread(heavy_cpu_inference, tensor)
```

---

## 5. Metaclasses & Class Construction (`__new__` vs `__init__`)

A metaclass customizes the creation of classes (instances of `type`):
```python
class StrictModelMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # Enforce that subclasses do not define mutable class attributes
        for attr, val in namespace.items():
            if not attr.startswith("__") and isinstance(val, (list, dict, set)):
                raise TypeError(f"Class attribute '{attr}' cannot be mutable. Use frozen types or instance attributes.")
        return super().__new__(mcs, name, bases, namespace)

class BaseModel(metaclass=StrictModelMeta):
    pass
```

---

## 6. Context Managers via `contextlib`

```python
from contextlib import contextmanager, ContextDecorator
import time

class PrecisionTimer(ContextDecorator):
    def __enter__(self):
        self.t0 = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter_ns() - self.t0) / 1_000_000.0
        return False  # Do not suppress exceptions
```
