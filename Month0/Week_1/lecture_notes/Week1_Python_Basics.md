Week 1: Programming Fundamentals & Python Basics

Learning Objectives

By the end of this week, you will be able to:
- [ ] Understand what programming and algorithms are
- [ ] Set up Python on your computer
- [ ] Write and run basic Python programs
- [ ] Use variables and understand data types
- [ ] Perform basic operations (arithmetic, string)
- [ ] Use the `print()` function to display output
- [ ] Debug simple programs

1. What is Programming?

Definition
Programming is giving a computer step-by-step instructions to solve a problem. Think of it like writing a recipe:
- A recipe tells a chef what to do
- A program tells a computer what to do

Key Concepts

Algorithm
An algorithm is a sequence of clear, unambiguous steps to solve a problem.

Example Algorithm: Make a Sandwich
```
1. Get two slices of bread
2. Open the peanut butter jar
3. Spread peanut butter on one slice
4. Open the jelly jar
5. Spread jelly on the other slice
6. Put the slices together
7. Cut diagonally (optional)
8. Enjoy!
```

Why Python?
Python is an excellent first language because:
- ✅ Readable: Code looks like English
- ✅ Beginner-friendly: Gentle learning curve
- ✅ Powerful: Used in real ML/AI projects
- ✅ Versatile: Web, data science, automation, etc.

2. Setting Up Python

Step 1: Install Python

macOS
1. Go to [python.org](https://www.python.org/)
2. Click "Downloads" → Choose macOS
3. Download Python 3.10+ installer
4. Run the installer and follow prompts
5. Open Terminal and type: `python3 --version`

Windows
1. Go to [python.org](https://www.python.org/)
2. Click "Downloads" → Choose Windows
3. Download Python 3.10+ installer
4. IMPORTANT: Check "Add Python to PATH" during installation
5. Open Command Prompt and type: `python --version`

Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
```

Step 2: Choose an Editor

Option 1: Terminal/Command Line (Simplest)
```bash
macOS/Linux: Open Terminal
Windows: Open Command Prompt

python3   # Start interactive Python
or
python3 filename.py   # Run a file
```

Option 2: IDLE (Built-in)
- Simple editor that comes with Python
- Good for beginners
- Search for "IDLE" on your computer

Option 3: VS Code (Recommended)
1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install Python extension
3. Create a new file `hello.py`
4. Write Python code
5. Run with Ctrl+F5

Option 4: Online (No Installation)
- [Replit.com](https://replit.com/) - Code in browser
- [Python Tutor](https://pythontutor.com/) - Visualize execution

Step 3: Your First Program

Create a file called `hello.py`:

```python
print("Hello, World!")
```

Run it:
```bash
python3 hello.py
```

Output:
```
Hello, World!
```

🎉 Congratulations! You've written your first program!

3. Python Basics

3.1 Variables: Storing Information

A variable is a container that holds a value. Think of it like a labeled box.

```python
Creating variables (assignment)
name = "Alice"
age = 25
height = 5.7
is_student = True

Displaying variables
print(name)      # Output: Alice
print(age)       # Output: 25
```

Variable Naming Rules:
- ✅ Start with letter or underscore: `myvar`, `count`, `name1`
- ❌ Don't start with numbers: `1name`
- ✅ Use underscores for spaces: `my_age` (not `myAge`)
- ✅ Case-sensitive: `name` and `Name` are different

Good Practice:
```python
Good variable names (clear and descriptive)
student_name = "Bob"
student_age = 20
gpa = 3.8

Avoid unclear names
x = "Bob"      # What is x?
sn = "Bob"     # What does sn mean?
```

3.2 Data Types: Types of Values

Python recognizes different types of data:

Integer (whole numbers)
```python
age = 25
count = 100
temperature = -5
```

Float (decimal numbers)
```python
height = 5.7
pi = 3.14159
price = 19.99
```

String (text)
```python
name = "Alice"
message = "Hello, Python!"
quote = 'Python is fun'  # Single or double quotes work

Multi-line strings
poem = """
Roses are red,
Violets are blue,
Python is awesome,
And so are you!
"""
```

Boolean (True or False)
```python
is_student = True
has_license = False
is_raining = True
```

3.3 Checking Data Types

Use the `type()` function:

```python
print(type(25))           # <class 'int'>
print(type(3.14))         # <class 'float'>
print(type("hello"))      # <class 'str'>
print(type(True))         # <class 'bool'>
```

4. Operations & Expressions

4.1 Arithmetic Operations

```python
Addition
result = 5 + 3           # 8

Subtraction
result = 10 - 4          # 6

Multiplication
result = 6 * 7           # 42

Division (always returns float)
result = 15 / 3          # 5.0

Floor Division (rounds down)
result = 17 // 5         # 3

Exponentiation (power)
result = 2  3          # 8 (2 to the power of 3)

Modulo (remainder)
result = 17 % 5          # 2 (remainder of 17 ÷ 5)
```

Real Example:
```python
Calculate total price with tax
item_price = 50
tax_rate = 0.08  # 8% tax
total = itemprice + (itemprice * tax_rate)
print(total)     # 54.0
```

4.2 String Operations

```python
Concatenation (joining strings)
first_name = "John"
last_name = "Doe"
fullname = firstname + " " + last_name
print(full_name)  # Output: John Doe

Repetition
laugh = "ha" * 3
print(laugh)      # Output: hahaha

Length
message = "Hello"
print(len(message))  # Output: 5
```

4.3 Order of Operations

Python follows standard math order (PEMDAS):

```python
result = 2 + 3 * 4       # 14 (not 20) - multiplication first
result = (2 + 3) * 4     # 20 - parentheses first
result = 10 / 2 * 3      # 15.0 - left to right
```

5. Type Conversion

Sometimes you need to convert between types:

```python
String to Integer
age_string = "25"
age = int(age_string)    # 25
print(type(age))         # <class 'int'>

String to Float
price_string = "19.99"
price = float(price_string)  # 19.99

Number to String
age = 25
age_string = str(age)    # "25"

String to Boolean (almost everything is True, except empty strings)
print(bool("hello"))     # True
print(bool(""))          # False
print(bool(0))           # False
print(bool(1))           # True
```

6. The `print()` Function

The `print()` function displays output:

```python
Simple output
print("Hello!")

Multiple values (separated by comma)
print("Name:", "Alice", "Age:", 25)
Output: Name: Alice Age: 25

Special characters
print("Line 1\nLine 2")   # \n = new line
print("Tab\there")        # \t = tab character

Formatting (f-strings) - Python 3.6+
name = "Bob"
age = 30
print(f"My name is {name} and I'm {age} years old")
Output: My name is Bob and I'm 30 years old
```

7. Getting Input from Users

Use `input()` to ask users for information:

```python
Get user input (always returns a string)
name = input("What's your name? ")
print("Hello, " + name + "!")

Convert to number
age_string = input("How old are you? ")
age = int(age_string)    # Convert string to integer
next_age = age + 1
print(f"Next year you'll be {next_age}")
```

Interactive Example:
```python
Calculator
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
sum_result = num1 + num2
print(f"{num1} + {num2} = {sum_result}")
```

8. Comments

Comments explain code but don't execute:

```python
This is a single-line comment

Variable for storing student name
name = "Alice"  # Name of the student

"""
This is a multi-line comment.
Use it for longer explanations.
Great for documentation!
"""
```

9. Common Pitfalls & Best Practices

❌ Mistake 1: Forgetting Quotes for Strings
```python
Wrong
print(Hello)         # Error: Hello is undefined

Right
print("Hello")       # Output: Hello
```

❌ Mistake 2: Mixing Up Operators
```python
Wrong (quotes in wrong place)
age = "25"
next_age = age + 1   # Error: can't add string + int

Right
age = 25
next_age = age + 1   # 26
```

✅ Best Practice 1: Clear Variable Names
```python
Bad
x = 100
y = 25

Good
student_name = "Alice"
student_score = 95
```

✅ Best Practice 2: Add Spaces Around Operators
```python
Harder to read
total=item_price+tax

Better
total = item_price + tax
```

10. Practice Code

Try running these examples:

Example 1: Simple Calculator
```python
Ask user for two numbers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

Calculate
add = num1 + num2
subtract = num1 - num2
multiply = num1 * num2
divide = num1 / num2

Display results
print(f"Addition: {add}")
print(f"Subtraction: {subtract}")
print(f"Multiplication: {multiply}")
print(f"Division: {divide}")
```

Example 2: Temperature Converter
```python
Convert Celsius to Fahrenheit
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")
```

Example 3: Personal Information
```python
Collect user info
name = input("What's your name? ")
age = int(input("How old are you? "))
city = input("Where do you live? ")

Display info
print(f"\n--- Your Profile ---")
print(f"Name: {name}")
print(f"Age: {age}")
print(f"City: {city}")
print(f"Born in: {2024 - age}")  # Approximate birth year
```

🎯 Key Takeaways

| Concept | Summary |
|---------|---------|
| Variables | Containers for storing values |
| Data Types | int, float, str, bool |
| Operations | +, -, , /, *, %, // |
| Strings | Text enclosed in quotes |
| Functions | `print()`, `input()`, `type()`, `len()` |
| Type Conversion | `int()`, `float()`, `str()`, `bool()` |

📚 Additional Resources

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python - Basics](https://realpython.com/python-basics/)
- [W3Schools Python](https://www.w3schools.com/python/)
- [Codecademy - Python Course](https://www.codecademy.com/learn/learn-python)

✅ Self-Assessment

Do you understand?
- [ ] What programming and algorithms are?
- [ ] How to create and run Python files?
- [ ] How to use variables and data types?
- [ ] How to perform arithmetic and string operations?
- [ ] How to use `print()` and `input()` functions?

If yes to all: Move to exercises! 
If no to some: Reread that section or post a question.

Ready for exercises? → [Week 1 Exercises](../exercises/Week1_Exercises.md)
