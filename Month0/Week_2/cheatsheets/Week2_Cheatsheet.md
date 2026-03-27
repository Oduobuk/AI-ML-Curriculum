Week 2: Control Flow & Functions Cheatsheet

Quick reference for control flow and functions.

📌 Conditional Statements

If/Elif/Else
```python
if condition:
    # code if True
elif other_condition:
    # code if elif True
else:
    # code if all False
```

Comparison Operators
```python
==    # Equal
!=    # Not equal
>     # Greater than
<     # Less than
>=    # Greater or equal
<=    # Less or equal
```

Boolean Operators
```python
and   # Both must be True
or    # At least one True
not   # Reverses condition
```

🔄 Loops

For Loop
```python
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for item in my_list:
    print(item)
```

Range Function
```python
range(5)        # 0, 1, 2, 3, 4
range(1, 5)     # 1, 2, 3, 4
range(0, 10, 2) # 0, 2, 4, 6, 8
range(5, 0, -1) # 5, 4, 3, 2, 1
```

While Loop
```python
while condition:
    # code
    # MUST update condition or infinite loop!
```

Loop Control
```python
break       # Exit loop immediately
continue    # Skip to next iteration
```

📊 Lists

Creating & Accessing
```python
my_list = [1, 2, 3, 4, 5]
my_list[0]      # 1 (first element)
my_list[-1]     # 5 (last element)
len(my_list)    # 5
```

List Methods
```python
my_list.append(6)           # Add to end
my_list.insert(0, 0)        # Insert at index
my_list.remove(3)           # Remove value
my_list.pop()               # Remove last
my_list.pop(0)              # Remove at index
my_list.sort()              # Sort in place
my_list.reverse()           # Reverse in place
my_list.count(2)            # Count occurrences
my_list.index(3)            # Find index of value
```

List Operations
```python
[1, 2] + [3, 4]  # [1, 2, 3, 4]
[5] * 3          # [5, 5, 5]
2 in [1, 2, 3]   # True
```

📖 Dictionaries

Creating & Accessing
```python
my_dict = {"name": "Alice", "age": 25}
my_dict["name"]          # "Alice"
my_dict["age"] = 26      # Update
my_dict["city"] = "NYC"  # Add new
del my_dict["age"]       # Delete key
```

Dictionary Methods
```python
my_dict.keys()       # All keys
my_dict.values()     # All values
my_dict.items()      # Key-value pairs
"name" in my_dict    # True if key exists
my_dict.get("name")  # Get value (None if not found)
```

Looping Through Dictionaries
```python
for key in my_dict:
    print(f"{key}: {my_dict[key]}")

for key, value in my_dict.items():
    print(f"{key}: {value}")
```

🔧 Functions

Basic Function
```python
def my_function():
    print("Hello")

my_function()  # Call it
```

With Parameters
```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

With Return Value
```python
def add(a, b):
    return a + b

result = add(5, 3)  # 8
```

Default Parameters
```python
def greet(name="Friend"):
    print(f"Hello, {name}!")

greet()           # Hello, Friend!
greet("Alice")    # Hello, Alice!
```

Multiple Return Values
```python
def get_info():
    return "Alice", 25, "NYC"

name, age, city = get_info()
```

Variable Scope
```python
x = 10  # Global

def my_function():
    x = 5  # Local - doesn't affect global
    print(x)  # 5

my_function()
print(x)  # 10 - global unchanged
```

🎯 Common Patterns

Count Items
```python
items = [1, 2, 2, 3, 3, 3]
count = items.count(3)  # 3
```

Find Maximum
```python
numbers = [3, 7, 2, 9, 1]
max_num = max(numbers)  # 9
```

Sum All
```python
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)  # 15
```

Filter List
```python
numbers = [1, 2, 3, 4, 5]
evens = [n for n in numbers if n % 2 == 0]  # [2, 4]
```

Convert Types
```python
lst = [1, 2, 3]
s = sum(lst) / len(lst)  # Average
```

🐛 Common Mistakes

| Mistake | Wrong | Right |
|---------|-------|-------|
| Missing colon | `if x > 5 print(x)` | `if x > 5: print(x)` |
| Forgot parentheses | `len mylist` | `len(mylist)` |
| Wrong equality | `if x = 5:` | `if x == 5:` |
| Index out of range | `list[5]` when 5 elements | Check length first |
| Return outside function | `return x` (not in function) | Must be inside function |
| Infinite loop | `while True: print(x)` | Add: `x += 1` |

[Back to Lecture](../lecturenotes/Week2ControlFlowFunctions.md)
