Week 1: Python Basics Cheatsheet

Quick reference guide for Week 1 concepts.

📌 Variables & Data Types

```python
Creating variables (assignment)
name = "Alice"              # String
age = 25                    # Integer
height = 5.7                # Float
is_student = True           # Boolean

Check data type
type(age)                   # <class 'int'>
type("hello")               # <class 'str'>
```

| Type | Example | Use Case |
|------|---------|----------|
| `int` | 42, -5, 0 | Whole numbers |
| `float` | 3.14, -2.5 | Decimals |
| `str` | "hello", 'world' | Text |
| `bool` | True, False | Yes/no values |

🔢 Arithmetic Operators

```python
Basic operations
5 + 3         # 8 (addition)
10 - 4        # 6 (subtraction)
6 * 7         # 42 (multiplication)
15 / 3        # 5.0 (division)
17 // 5       # 3 (floor division)
17 % 5        # 2 (modulo/remainder)
2  3        # 8 (exponent/power)
```

📝 String Operations

```python
Concatenation (joining)
"Hello" + " " + "World"     # "Hello World"

Repetition
"Ha" * 3                    # "HaHaHa"

Length
len("Python")               # 6

Case conversion
"hello".upper()             # "HELLO"
"HELLO".lower()             # "hello"

Access individual character
text = "Python"
text[0]                     # "P" (first character)
text[-1]                    # "n" (last character)
```

🖨️ Output: `print()` Function

```python
Simple output
print("Hello, World!")

Multiple values
print("Name:", "Alice", "Age:", 25)

Special characters
print("Line 1\nLine 2")     # \n = newline
print("Tab\there")         # \t = tab

F-strings (Python 3.6+) - RECOMMENDED
name = "Bob"
age = 30
print(f"My name is {name} and I'm {age}")

Formatting with f-strings
price = 19.99
print(f"Price: ${price:.2f}")    # 2 decimal places
print(f"Name: {name:<20}")       # Left align, 20 chars
print(f"Age: {age:>5}")          # Right align, 5 chars
```

⌨️ Input: `input()` Function

```python
Get user input (always returns string)
name = input("Enter your name: ")

Convert to number
age = int(input("Enter your age: "))
height = float(input("Enter your height: "))
```

🔄 Type Conversion

```python
String to Integer
int("42")                   # 42
int("3.14")                 # Error! Can't convert with decimal

String to Float
float("3.14")               # 3.14
float("42")                 # 42.0

Number to String
str(42)                     # "42"
str(3.14)                   # "3.14"

To Boolean
bool(1)                     # True
bool(0)                     # False
bool("hello")               # True
bool("")                    # False (empty string)
```

💾 Variable Naming Rules

✅ Good Names:
```python
student_name = "Alice"
birth_year = 2000
heightincm = 170
is_valid = True
```

❌ Bad Names:
```python
x = "Alice"                 # Unclear
sn = "Alice"                # Confusing abbreviation
123name = "Alice"           # Starts with number
student name = "Alice"      # Contains space (invalid!)
```

🎯 Order of Operations (PEMDAS)

```python
Parentheses first
result = (2 + 3) * 4        # 20

Then Exponents
result = 2 + 3  2         # 11 (3² first, then add 2)

Then *, /, //, %
result = 2 + 3  4          # 14 (34 first, then add 2)

Left to right for same precedence
result = 10 / 2 * 5         # 25.0 (left to right)
```

💬 Comments

```python
Single line comment

"""
Multi-line comment
for longer explanations
"""

x = 5  # Comment at end of line
```

🐛 Common Mistakes

| Mistake | Wrong | Right |
|---------|-------|-------|
| Forgetting quotes | `print(Hello)` | `print("Hello")` |
| Wrong operator | `"a" + 5` | `"a" + str(5)` |
| No parentheses for function | `print "Hello"` | `print("Hello")` |
| String + number | `age + 5` (if age is "25") | `int(age) + 5` |

📋 Useful Functions Summary

| Function | Purpose | Example |
|----------|---------|---------|
| `print()` | Display output | `print("Hello")` |
| `input()` | Get user input | `name = input("Name: ")` |
| `int()` | Convert to integer | `int("42")` |
| `float()` | Convert to float | `float("3.14")` |
| `str()` | Convert to string | `str(42)` |
| `type()` | Check data type | `type(42)` |
| `len()` | Get length | `len("hello")` |
| `bool()` | Convert to boolean | `bool(1)` |

🚀 Quick Program Template

```python
Step 1: Get input
name = input("Enter your name: ")
age = int(input("Enter your age: "))

Step 2: Process/Calculate
next_age = age + 1

Step 3: Display output
print(f"{name} will be {next_age} next year!")
```

📊 Common Patterns

Temperature Conversion
```python
celsius = float(input("Temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")
```

Area Calculation
```python
length = float(input("Length: "))
width = float(input("Width: "))
area = length * width
print(f"Area: {area} square units")
```

BMI Calculation
```python
weight = float(input("Weight (lbs): "))
height = float(input("Height (inches): "))
bmi = (weight / (height  2)) * 703
print(f"BMI: {bmi:.2f}")
```

🎓 Key Concepts to Remember

- Variables store information
- Data types determine what operations are valid
- Functions perform actions (print, input, etc.)
- Type conversion changes data types
- F-strings format output nicely
- Comments explain your code

Need more help? Review the [Week 1 Lecture Notes](../lecturenotes/Week1Python_Basics.md)
