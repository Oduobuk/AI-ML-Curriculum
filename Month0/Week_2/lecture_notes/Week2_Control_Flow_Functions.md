Week 2: Control Flow & Functions

Learning Objectives

By the end of this week, you will be able to:
- [ ] Use if/elif/else statements to make decisions
- [ ] Understand boolean logic (and, or, not)
- [ ] Write loops (for, while) to repeat code
- [ ] Use loop control statements (break, continue)
- [ ] Create and use functions with parameters and return values
- [ ] Understand variable scope
- [ ] Organize code into reusable functions

1. Conditional Statements: Making Decisions

1.1 The `if` Statement

Programs need to make decisions based on conditions. That's where `if` comes in.

```python
Basic if statement
age = 18
if age >= 18:
    print("You are an adult")

Without if, the code always runs
print("This always prints")
```

Key Points:
- `:` (colon) ends the if line
- Indentation (4 spaces) shows what code belongs to the if block
- The code indented after `if` only runs if condition is True

1.2 The `if/else` Statement

What if you want to do something when the condition is False?

```python
age = 15
if age >= 18:
    print("You can vote")
else:
    print("You cannot vote yet")
```

1.3 Multiple Conditions: `if/elif/else`

Use `elif` (else if) for multiple conditions:

```python
score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: F")
```

Important: Python checks conditions from top to bottom and stops at the first True one.

1.4 Comparison Operators

```python
Equals
5 == 5          # True
5 == 3          # False

Not equals
5 != 3          # True
5 != 5          # False

Greater than / Less than
10 > 5          # True
10 < 5          # False
10 >= 10        # True
10 <= 5         # False
```

⚠️ Common Mistake: `=` is assignment, `==` is comparison!

```python
Wrong - assigns value
if age = 18:    # SyntaxError!
    print("Adult")

Right - compares
if age == 18:   # Correct
    print("Adult")
```

1.5 Boolean Logic

Combine conditions with `and`, `or`, `not`:

```python
AND: both must be True
age = 20
has_license = True

if age >= 18 and has_license:
    print("Can drive")  # Prints because both are True

OR: at least one must be True
day = "Saturday"
hasfreetime = True

if day == "Saturday" or day == "Sunday":
    print("It's the weekend")  # Prints

NOT: reverses the condition
is_raining = False

if not is_raining:
    print("Let's go outside")  # Prints
```

Truth Tables:

```
AND (both must be True):
True and True   → True
True and False  → False
False and True  → False
False and False → False

OR (at least one True):
True or True    → True
True or False   → True
False or True   → True
False or False  → False

NOT (reverses):
not True        → False
not False       → True
```

2. Loops: Repeating Code

2.1 `for` Loops: Loop a Set Number of Times

```python
Print 1 through 5
for i in range(1, 6):
    print(i)
Output: 1 2 3 4 5

range(start, stop, step)
Note: stop is NOT included!
```

What is `range()`?

```python
range(5)           # 0, 1, 2, 3, 4 (starts at 0)
range(1, 5)        # 1, 2, 3, 4 (1 to 4)
range(0, 10, 2)    # 0, 2, 4, 6, 8 (every 2nd number)
range(10, 0, -1)   # 10, 9, 8, ..., 1 (counting down)
```

2.2 Loop Through Lists

```python
fruits = ["apple", "banana", "orange"]

Method 1: Loop through directly
for fruit in fruits:
    print(fruit)
Output:
apple
banana
orange

Method 2: Loop with index
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")
Output:
0: apple
1: banana
2: orange
```

2.3 `while` Loops: Loop Until a Condition is False

```python
Count down
count = 5
while count > 0:
    print(count)
    count = count - 1  # Decrease count each iteration
print("Blastoff!")
```

⚠️ Be Careful: An infinite loop never ends!

```python
Don't do this:
while True:
    print("This never stops!")  # Infinite loop!
```

2.4 Loop Control: `break` and `continue`

`break`: Exit the loop immediately

```python
for i in range(1, 10):
    if i == 5:
        break           # Stop when i equals 5
    print(i)
Output: 1 2 3 4
```

`continue`: Skip to next iteration

```python
for i in range(1, 6):
    if i == 3:
        continue        # Skip when i equals 3
    print(i)
Output: 1 2 4 5
```

3. Data Structures: Lists and Dictionaries

3.1 Lists: Ordered Collections

```python
Create a list
numbers = [10, 20, 30, 40]
names = ["Alice", "Bob", "Charlie"]

Access elements (index starts at 0)
print(names[0])          # "Alice"
print(names[1])          # "Bob"
print(names[-1])         # "Charlie" (last element)

Modify elements
names[0] = "Alfred"

Add elements
names.append("Diana")    # Add to end
names.insert(1, "Betty") # Insert at position 1

Remove elements
names.pop()              # Remove last element
names.remove("Betty")    # Remove specific value

List length
len(names)               # Number of elements

Check if element exists
"Alice" in names         # True or False

List operations
combined = [1, 2] + [3, 4]  # [1, 2, 3, 4]
repeated = [5] * 3          # [5, 5, 5]
```

3.2 Dictionaries: Key-Value Pairs

```python
Create a dictionary
student = {
    "name": "Alice",
    "age": 20,
    "gpa": 3.8
}

Access values by key
print(student["name"])   # "Alice"
print(student["age"])    # 20

Modify values
student["age"] = 21

Add new key-value pair
student["major"] = "Computer Science"

Remove key-value pair
del student["gpa"]

Get all keys
print(student.keys())    # ['name', 'age', 'major', ...]

Get all values
print(student.values())  # ['Alice', 21, 'Computer Science', ...]

Check if key exists
if "name" in student:
    print(student["name"])

Loop through dictionary
for key in student:
    print(f"{key}: {student[key]}")
```

4. Functions: Reusable Code

Functions let you write code once and use it many times.

4.1 Basic Function Structure

```python
def greet():
    print("Hello, World!")

Call the function
greet()
greet()  # You can call it multiple times
```

4.2 Functions with Parameters

Parameters let functions work with different data:

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")   # Hello, Alice!
greet("Bob")     # Hello, Bob!
```

Multiple Parameters:

```python
def add(num1, num2):
    result = num1 + num2
    print(f"{num1} + {num2} = {result}")

add(5, 3)        # 5 + 3 = 8
add(10, 20)      # 10 + 20 = 30
```

4.3 Return Values

Functions can return values instead of just printing:

```python
def multiply(a, b):
    return a * b

result = multiply(4, 5)
print(result)     # 20

Use return value directly
total = multiply(3, 7) + multiply(2, 6)
print(total)      # 33
```

4.4 Default Parameters

Set default values for parameters:

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")              # Hello, Alice!
greet("Bob", "Hi")          # Hi, Bob!
greet("Charlie", "Hey")     # Hey, Charlie!
```

4.5 Variable Scope

Variables created inside functions are local (only exist in that function):

```python
def set_age():
    age = 25  # Local variable
    print(age)

set_age()      # 25
print(age)     # Error! age doesn't exist here
```

Global Variables:

```python
age = 30  # Global variable

def print_age():
    print(age)  # Can read global variables

print_age()    # 30
```

5. Practical Examples

Example 1: Password Validator

```python
def isvalidpassword(password):
    if len(password) < 8:
        return False
    if password.isdigit():  # Only numbers
        return False
    if password.isalpha():  # Only letters
        return False
    return True

pwd = input("Enter password: ")
if isvalidpassword(pwd):
    print("Password is strong")
else:
    print("Password is weak")
```

Example 2: Average Calculator

```python
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = sum(numbers)
    return total / len(numbers)

scores = [85, 90, 78, 92]
avg = calculate_average(scores)
print(f"Average: {avg:.2f}")
```

Example 3: Inventory System

```python
inventory = {
    "apples": 10,
    "bananas": 5,
    "oranges": 8
}

def additem(itemname, quantity):
    if item_name in inventory:
        inventory[item_name] += quantity
    else:
        inventory[item_name] = quantity

def sellitem(itemname, quantity):
    if item_name in inventory:
        if inventory[item_name] >= quantity:
            inventory[item_name] -= quantity
            return True
    return False

add_item("apples", 5)
sell_item("bananas", 2)
print(inventory)
```

6. Common Pitfalls

❌ Mistake 1: Indentation Error
```python
Wrong
if x > 5:
print("x is big")  # Not indented!

Right
if x > 5:
    print("x is big")  # Indented
```

❌ Mistake 2: Infinite Loop
```python
Wrong
while True:
    print("Never stops!")  # Infinite loop

Right
count = 0
while count < 5:
    print(count)
    count += 1  # Eventually becomes >= 5
```

❌ Mistake 3: Index Out of Range
```python
names = ["Alice", "Bob"]
print(names[5])  # Error! Only indices 0-1 exist
```

✅ Best Practices

- Write small functions that do one thing
- Use meaningful names for functions and variables
- Add comments to explain complex logic
- Test edge cases (empty lists, zero values, etc.)

🎯 Key Takeaways

| Concept | Purpose | Example |
|---------|---------|---------|
| `if/elif/else` | Make decisions | `if age >= 18:` |
| Comparison operators | Compare values | `x == 5`, `x > 10` |
| Boolean logic | Combine conditions | `x > 5 and y < 10` |
| `for` loop | Loop known times | `for i in range(10):` |
| `while` loop | Loop until condition | `while count > 0:` |
| Lists | Ordered collection | `[1, 2, 3]` |
| Dictionaries | Key-value pairs | `{"name": "Alice"}` |
| Functions | Reusable code | `def my_func():` |
| Parameters | Function inputs | `def add(a, b):` |
| Return | Function output | `return result` |

📚 Additional Resources

- [Python Documentation: Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [Real Python: Loops](https://realpython.com/loops/)
- [Real Python: Functions](https://realpython.com/defining-your-own-python-function/)

Ready for exercises? → [Week 2 Exercises](../exercises/Week2_Exercises.md)
