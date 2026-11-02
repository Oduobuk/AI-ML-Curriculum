# Month 0, Week 1: Modern Python Toolchain & Programming Fundamentals

## 1. The Modern AI Engineer's Toolkit (2026 Edition)

Traditional data science tutorials often begin by instructing students to install heavyweight distributions like Anaconda (several gigabytes of outdated packages). Modern AI engineering prioritizes **speed, minimal overhead, and exact reproducibility**.

### The Fast Python Toolchain: `uv`
Developed by Astral, `uv` is an extremely fast Python package and project manager written in Rust. It replaces `pip`, `pip-tools`, `poetry`, and `pyenv`.

```bash
# 1. Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Verify installation
uv --version

# 3. Create a clean isolated virtual environment
uv venv .venv --python 3.11

# 4. Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 5. Fast install packages
uv pip install numpy pydantic ruff
```

---

## 2. Variables, Memory References & Primitive Types

In Python, variables do not store values directly; they store **references** (pointers) to objects in memory.

### The 4 Core Primitives:
```python
# 1. Integer (int): Arbitrary precision whole numbers
dataset_size: int = 150_000
batch_size: int = 64

# 2. Floating-point (float): 64-bit double precision numbers
learning_rate: float = 0.0003
accuracy: float = 98.75

# 3. String (str): Immutable Unicode text
model_architecture: str = "Transformer-Decoder"

# 4. Boolean (bool): Truth values (True / False)
is_cuda_available: bool = True
use_flash_attention: bool = False
```

### Checking Types and Object IDs:
```python
x = 42
print(type(x))       # <class 'int'>
print(id(x))         # Memory address of object 42

# Python optimizes small integers via integer caching:
a = 256
b = 256
print(a is b)        # True (both point to identical cached memory location)
```

---

## 3. Operations & Type Conversions

### Arithmetic Operators:
| Operator | Name | Example | Result |
| :--- | :--- | :--- | :--- |
| `+` | Addition | `10 + 5` | `15` |
| `-` | Subtraction | `10 - 5` | `5` |
| `*` | Multiplication | `10 * 5` | `50` |
| `/` | True Division (Float) | `10 / 4` | `2.5` |
| `//` | Floor Division (Integer) | `10 // 4` | `2` |
| `%` | Modulo (Remainder) | `10 % 3` | `1` |
| `**` | Exponentiation | `2 ** 8` | `256` |

### Explicit Type Casting:
```python
raw_input_epochs = "25"
epochs = int(raw_input_epochs)    # Converts string to integer

loss_score = 0.04592
loss_str = f"{loss_score:.3f}"    # Formatted f-string: '0.046'
```

---

## 4. Modern String Formatting: F-Strings
F-strings (formatted string literals) provide concise, readable string interpolation:

```python
epoch = 5
total_epochs = 20
train_loss = 0.12459

# Inline expression evaluation and formatting
print(f"Epoch [{epoch:02d}/{total_epochs:02d}] - Loss: {train_loss:.4f}")
# Output: Epoch [05/20] - Loss: 0.1246
```

---

## 5. Defensive Programming & Code Quality with Ruff

Modern teams enforce code quality through automated linters:
```bash
# Check code for common bugs and formatting issues
uv run ruff check .

# Automatically fix format and lint errors
uv run ruff check --fix .
uv run ruff format .
```

---

## Hands-On Coding Assignment (Preview)
Navigate to [Week 1 Assignment](../assignments/Week1_Assignment.md) to implement your first production-style Python CLI script with parameter verification.
