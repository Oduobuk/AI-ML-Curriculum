# Month 0, Week 2: Control Flow, Functions & Structured Data Schemas

## 1. Control Flow & Logical Evaluation

Every algorithm requires decision paths and iterative loops.

### Conditional Branching:
```python
def classify_latency(latency_ms: float) -> str:
    """Classifies model inference latency based on service SLAs."""
    if latency_ms < 50:
        return "Ultra-fast (Ideal for Real-Time Streaming)"
    elif 50 <= latency_ms < 200:
        return "Acceptable (Standard Interactive Chat)"
    else:
        return "Slow (Requires Caching or Quantization)"

print(classify_latency(32.4))
```

### Iteration & Comprehensions:
Avoid verbose loops where functional comprehensions produce cleaner, faster code:

```python
# Bad: Procedural loop with append
squared_errors = []
for error in [0.2, 0.5, 1.2]:
    squared_errors.append(error ** 2)

# Good: Idiomatic List Comprehension
squared_errors = [e ** 2 for e in [0.2, 0.5, 1.2]]

# Dict Comprehension
labels = ["cat", "dog", "car"]
label_to_id = {label: idx for idx, label in enumerate(labels)}
# Result: {'cat': 0, 'dog': 1, 'car': 2}
```

---

## 2. Robust Functions & Type Hinting

Clean functions have clear responsibilities, documented arguments, and typed signatures.

```python
from typing import List, Optional, Tuple

def compute_batch_statistics(
    losses: List[float], 
    threshold: Optional[float] = None
) -> Tuple[float, float, int]:
    """Computes mean loss, max loss, and anomalies exceeding an optional threshold.
    
    Args:
        losses: List of float loss values recorded across batches.
        threshold: Optional cutoff for anomaly counting.
        
    Returns:
        A tuple of (mean_loss, max_loss, anomaly_count).
    """
    if not losses:
        raise ValueError("The losses list cannot be empty.")
        
    mean_val = sum(losses) / len(losses)
    max_val = max(losses)
    anomalies = sum(1 for l in losses if threshold is not None and l > threshold)
    
    return mean_val, max_val, anomalies
```

---

## 3. Structured Data Validation with Pydantic v2

In AI Engineering, handling unstructured API responses and LLM payloads using unverified Python dictionaries causes silent runtime crashes. **Pydantic** is the gold standard for runtime schema enforcement.

```python
from pydantic import BaseModel, Field, field_validator
from typing import List

class ModelConfig(BaseModel):
    model_name: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    max_tokens: int = Field(gt=0, le=8192, default=512)
    stop_sequences: List[str] = Field(default_factory=list)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        valid_models = ["llama-3.2-3b", "qwen2.5-coder-7b", "mistral-7b"]
        if v.lower() not in valid_models:
            raise ValueError(f"Unsupported model '{v}'. Must be one of {valid_models}")
        return v.lower()

# Valid Input:
config = ModelConfig(model_name="llama-3.2-3b", temperature=0.5)
print(config.model_dump_json(indent=2))

# Invalid Input (Throws clean ValidationError):
try:
    invalid = ModelConfig(model_name="gpt-legacy", temperature=3.5)
except Exception as e:
    print(f"Validation Caught Error:\n{e}")
```

---

## 4. Error Handling (`try / except / finally`)

Gracefully intercept network errors, bad files, or division errors:

```python
def safe_divide(numerator: float, denominator: float) -> float:
    try:
        return numerator / denominator
    except ZeroDivisionError:
        print("Warning: Zero denominator encountered. Returning 0.0.")
        return 0.0
    except TypeError as err:
        print(f"Invalid types provided: {err}")
        raise
```

---

## Practice & Assignments
* Complete [Week 2 Exercises](../exercises/Week2_Exercises.md)
* Build the [Week 2 Assignment](../assignments/Week2_Assignment.md)
