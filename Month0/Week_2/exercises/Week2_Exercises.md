# Week 2: Graduate-Level Problem Set — Advanced Control Flow, Metaprogramming & Asynchronous Systems

Welcome to the Week 2 Problem Set. This suite of problems advances beyond basic syntax into the mechanics of Python's execution model: closure cell objects, decorator stacking with metadata preservation, custom generator dispatch loops, cooperative asynchronous concurrency with task scheduling, and metaclass-driven schema validation.

---

## Tier 1: Core Mechanics & Bytecode Inspection

### Exercise 1: Closure Cell Inspection & Late-Binding Resolution
**Concepts**: Lexical closures, `__closure__`, `cell_contents`, late-binding variable resolution in loops.

#### Problem Specification
Consider the classic late-binding trap where factory functions created inside a loop bind to the enclosing scope's loop variable rather than capturing the immediate value:
```python
def make_multipliers_faulty():
    return [lambda x: i * x for i in range(4)]
```
1. Inspect the `__closure__` attribute and `cell_contents` of the resulting functions to prove why all instances evaluate using `i = 3`.
2. Write `make_multipliers_fixed()` using two distinct idiomatic mechanisms:
   - Default argument capture (`lambda x, i=i: ...`).
   - Explicit closure factory wrapping (`(lambda i: lambda x: i * x)(i)`).
3. Write an inspection function `inspect_closure_cells(fn)` that returns the memory addresses and values of all closed-over cells.

#### Verification Solution
```python
import inspect
from typing import Callable, List, Any, Tuple

def make_multipliers_faulty() -> List[Callable[[int], int]]:
    return [lambda x: i * x for i in range(4)]

def make_multipliers_default_arg() -> List[Callable[[int], int]]:
    return [lambda x, i=i: i * x for i in range(4)]

def make_multipliers_factory() -> List[Callable[[int], int]]:
    def factory(i: int) -> Callable[[int], int]:
        return lambda x: i * x
    return [factory(i) for i in range(4)]

def inspect_closure_cells(fn: Callable[..., Any]) -> List[Tuple[int, Any]]:
    if not fn.__closure__:
        return []
    return [(id(cell), cell.cell_contents) for cell in fn.__closure__]

# Verification Run
if __name__ == "__main__":
    faulty = make_multipliers_faulty()
    print("Faulty evaluations for x=2:", [f(2) for f in faulty])  # [6, 6, 6, 6]
    print("Faulty closure cell IDs:", [inspect_closure_cells(f) for f in faulty])
    
    fixed_default = make_multipliers_default_arg()
    print("Fixed (default arg) for x=2:", [f(2) for f in fixed_default])  # [0, 2, 4, 6]
    
    fixed_factory = make_multipliers_factory()
    print("Fixed (factory) for x=2:", [f(2) for f in fixed_factory])  # [0, 2, 4, 6]
    print("Factory closure cells:", [inspect_closure_cells(f) for f in fixed_factory])
```

---

### Exercise 2: Parameterized Decorator with Introspection & Wrapped Signature Preservation
**Concepts**: `@functools.wraps`, `inspect.signature`, AST decorator chaining order.

#### Problem Specification
Construct a decorator `@audit_invocation(max_retries: int = 3, log_arguments: bool = True)` that:
1. Validates incoming arguments against the wrapped function's signature before invocation using `inspect.Signature.bind`.
2. Attempts execution up to `max_retries` on designated exceptions.
3. Completely preserves `__name__`, `__doc__`, `__annotations__`, and `__wrapped__` references.
4. Correctly exposes the unwrapped original function via `.get_original()`.

#### Verification Solution
```python
import functools
import inspect
import time
from typing import Callable, Any, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def audit_invocation(max_retries: int = 3, log_arguments: bool = True):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if log_arguments:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                # Production structured telemetry logging hook
                # print(f"[AUDIT] Invoking {func.__name__} with {bound_args.arguments}")
            
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        raise RuntimeError(f"{func.__name__} failed after {max_retries} attempts") from exc
                    time.sleep(0.01 * attempt)
            raise last_exc  # Unreachable but type-safe
        
        # Expose introspection attribute
        wrapper.get_original = lambda: getattr(func, "get_original", lambda: func)()  # type: ignore
        return wrapper
    return decorator

# Verification Run
if __name__ == "__main__":
    attempt_counter = 0

    @audit_invocation(max_retries=3, log_arguments=True)
    def unstable_network_call(endpoint: str, timeout: float = 1.0) -> str:
        """Fetch payload from simulated telemetry gateway."""
        global attempt_counter
        attempt_counter += 1
        if attempt_counter < 3:
            raise ConnectionResetError(f"Handshake timeout at {endpoint}")
        return f"200 OK from {endpoint}"

    assert unstable_network_call.__name__ == "unstable_network_call"
    assert unstable_network_call.__doc__ == "Fetch payload from simulated telemetry gateway."
    res = unstable_network_call("https://api.internal/v1")
    print("Execution output:", res)
    assert res == "200 OK from https://api.internal/v1"
    print("Unwrapped raw function:", unstable_network_call.get_original())  # type: ignore
```

---

## Tier 2: Intermediate Systems & Generators

### Exercise 3: Bidirectional Generator Protocol (Coroutines via `send` and `throw`)
**Concepts**: Python generators as state machines, `.send()`, `.throw()`, yield expressions.

#### Problem Specification
Build a streaming exponentially-weighted moving average (EWMA) engine `ewma_coroutine(alpha: float)` utilizing pure generator coroutine mechanics:
1. Must yield the current computed smoothed statistic.
2. Accepts new observation scalar values via `generator.send(value)`.
3. Handles `generator.throw(ResetSmootherException)` by resetting its running mean to the next incoming value without terminating the generator loop.
4. Returns total observations processed upon exit (`StopIteration.value`).

#### Verification Solution
```python
from typing import Generator

class ResetSmootherException(Exception):
    """Exception signal to reset internal EWMA accumulator."""
    pass

def ewma_coroutine(alpha: float) -> Generator[float, float, int]:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("Alpha smoothing factor must be in (0, 1].")
    
    total_samples = 0
    s_t = yield 0.0  # Prime coroutine and wait for first observation
    total_samples += 1
    
    while True:
        try:
            val = yield s_t
            total_samples += 1
            s_t = alpha * val + (1.0 - alpha) * s_t
        except ResetSmootherException:
            # Catch signal, reset internal state on next transmission
            val = yield s_t
            s_t = val
            total_samples += 1

# Verification Run
if __name__ == "__main__":
    smoother = ewma_coroutine(alpha=0.5)
    next(smoother)  # Prime the coroutine
    
    print("Sample 10.0 -> EWMA:", smoother.send(10.0))  # 10.0
    print("Sample 20.0 -> EWMA:", smoother.send(20.0))  # 0.5 * 20 + 0.5 * 10 = 15.0
    print("Sample 30.0 -> EWMA:", smoother.send(30.0))  # 0.5 * 30 + 0.5 * 15 = 22.5
    
    # Trigger state reset
    smoother.throw(ResetSmootherException)
    print("After Reset, Sample 100.0 -> EWMA:", smoother.send(100.0))  # Resets to 100.0
```

---

### Exercise 4: Context Managers via ContextDecorator & Generator Protocols
**Concepts**: `contextlib.contextmanager`, `__enter__`, `__exit__`, re-entrancy, performance timing.

#### Problem Specification
Implement a high-precision performance profiler `HardwareTimer` that operates both as a context manager (`with HardwareTimer() as t:`) and as a function decorator (`@HardwareTimer()`):
1. Records wall-clock time (`time.perf_counter_ns`) and CPU time (`time.process_time_ns`).
2. Provides access to `elapsed_wall_ms` and `elapsed_cpu_ms`.
3. Employs `contextlib.ContextDecorator` mechanics so it can decorate sync functions without overhead.

#### Verification Solution
```python
import time
from contextlib import ContextDecorator
from typing import Optional, Type

class HardwareTimer(ContextDecorator):
    def __init__(self, label: str = "Execution"):
        self.label = label
        self.start_wall: int = 0
        self.end_wall: int = 0
        self.start_cpu: int = 0
        self.end_cpu: int = 0
        self.elapsed_wall_ms: float = 0.0
        self.elapsed_cpu_ms: float = 0.0

    def __enter__(self) -> "HardwareTimer":
        self.start_cpu = time.process_time_ns()
        self.start_wall = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> bool:
        self.end_wall = time.perf_counter_ns()
        self.end_cpu = time.process_time_ns()
        self.elapsed_wall_ms = (self.end_wall - self.start_wall) / 1_000_000.0
        self.elapsed_cpu_ms = (self.end_cpu - self.start_cpu) / 1_000_000.0
        # Do not suppress exceptions
        return False

# Verification Run
if __name__ == "__main__":
    with HardwareTimer("Matrix Multiplication") as timer:
        total = sum(i * i for i in range(500_000))
    
    print(f"Wall time: {timer.elapsed_wall_ms:.4f} ms | CPU time: {timer.elapsed_cpu_ms:.4f} ms")
    assert timer.elapsed_wall_ms > 0
```

---

## Tier 3: Advanced Systems & Metaprogramming

### Exercise 5: Metaclass Attribute Enforcement & Type Validation Registry
**Concepts**: `type.__new__`, class attribute registration, validation invariants.

#### Problem Specification
Create a metaclass `ValidatedModelMeta` that enforces strict schema declaration on all subclasses:
1. Disallows mutable class variables (e.g., bare lists or dictionaries declared at class level).
2. Requires that all class attributes with type annotations define a `validate(value)` contract or default to primitive types.
3. Auto-registers all derived models into an immutable global `MODEL_REGISTRY: Dict[str, Type]`.

#### Verification Solution
```python
from typing import Dict, Type, Any

MODEL_REGISTRY: Dict[str, Type[Any]] = {}

class ValidatedModelMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # 1. Enforce no mutable class-level container variables
        for attr, value in namespace.items():
            if not attr.startswith("__"):
                if isinstance(value, (list, dict, set)):
                    raise TypeError(
                        f"Mutable class attribute '{attr}' on {name} is forbidden. "
                        "Define instance variables in __init__ or use frozen immutable defaults."
                    )
        
        # 2. Build class object
        cls = super().__new__(mcs, name, bases, namespace)
        
        # 3. Register class in global registry (excluding abstract base)
        if name != "BaseModel":
            if name in MODEL_REGISTRY:
                raise ValueError(f"Duplicate model definition for '{name}'")
            MODEL_REGISTRY[name] = cls
            
        return cls

class BaseModel(metaclass=ValidatedModelMeta):
    pass

# Verification Run
if __name__ == "__main__":
    class UserFeatureVector(BaseModel):
        dimension: int = 128
        embedding_name: str = "text-embedding-3-small"

    print("Registered models:", list(MODEL_REGISTRY.keys()))
    assert "UserFeatureVector" in MODEL_REGISTRY

    try:
        class FaultyModel(BaseModel):
            tags: list = []  # Should trigger TypeError
    except TypeError as e:
        print("Successfully intercepted mutable attribute error:", e)
```

---

### Exercise 6: Micro-Asyncio Event Loop from Scratch (Pure Python Generator Trampoline)
**Concepts**: Non-blocking IO event dispatching, coroutine scheduling, generator trampolining.

#### Problem Specification
Build a minimal, pure Python cooperative event loop `ToyAsyncEngine` capable of concurrently scheduling multiple generators yielding non-blocking sleep operations without using `asyncio`:
1. Implement `async_sleep(ticks: int)`.
2. Implement `spawn(coroutine)` to add coroutines into an active run-queue (`collections.deque`).
3. Run the trampoline loop until all tasks yield `StopIteration`.

#### Verification Solution
```python
from collections import deque
from typing import Generator, Any, Deque

class AsyncSleep:
    def __init__(self, ticks: int):
        self.remaining_ticks = ticks

class ToyAsyncEngine:
    def __init__(self):
        self.task_queue: Deque[Generator[Any, None, Any]] = deque()
        self.sleeping_tasks: list[tuple[int, Generator[Any, None, Any]]] = []
        self.tick_counter: int = 0

    def spawn(self, coro: Generator[Any, None, Any]):
        self.task_queue.append(coro)

    def run(self):
        while self.task_queue or self.sleeping_tasks:
            self.tick_counter += 1
            
            # Wake sleeping tasks
            still_sleeping = []
            for wake_tick, coro in self.sleeping_tasks:
                if self.tick_counter >= wake_tick:
                    self.task_queue.append(coro)
                else:
                    still_sleeping.append((wake_tick, coro))
            self.sleeping_tasks = still_sleeping

            if not self.task_queue:
                continue

            coro = self.task_queue.popleft()
            try:
                signal = coro.send(None)
                if isinstance(signal, AsyncSleep):
                    self.sleeping_tasks.append((self.tick_counter + signal.remaining_ticks, coro))
                else:
                    self.task_queue.append(coro)
            except StopIteration:
                pass  # Task completed successfully

# Verification Run
def task_worker(task_id: str, duration: int) -> Generator[Any, None, None]:
    print(f"Task {task_id}: Started")
    yield AsyncSleep(duration)
    print(f"Task {task_id}: Resumed after {duration} ticks")
    yield AsyncSleep(1)
    print(f"Task {task_id}: Finished")

if __name__ == "__main__":
    engine = ToyAsyncEngine()
    engine.spawn(task_worker("Alpha", 3))
    engine.spawn(task_worker("Beta", 1))
    print("Starting micro-event loop execution:")
    engine.run()
```
