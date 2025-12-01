
# Week 2: Python Fundamentals for AI/ML

## 1. Why Python for AI/ML?

Python is the world's fastest-growing and most popular programming language, not just for software developers, but also for data analysts, scientists, and AI engineers. Here’s why it's the ideal choice for data-centric fields:

*   **Simplicity and Readability:** Python's syntax is clean and intuitive, resembling plain English. This allows you to solve complex problems with fewer lines of code compared to other languages like C# or Java.
*   **Multi-Purpose:** It's a versatile language used for a wide range of tasks, including data analysis, AI and machine learning, web development, and automation.
*   **High-Level Language:** Python handles complex tasks like memory management automatically, so you can focus on solving problems rather than low-level details.
*   **Massive Ecosystem:** Python has a huge collection of libraries, frameworks, and tools. For AI and data science, libraries like **Pandas**, **Matplotlib**, and **Scikit-learn** provide powerful, pre-built functionality that dramatically speeds up development.
*   **Large Community:** With a massive global community, you can almost always find help and resources online when you get stuck.

## 2. Setting Up Your Environment

There are two primary ways to run Python code. For this course, we'll focus on the easiest method to get you started immediately.

*   **Local Installation:** This involves installing Python directly on your computer (Mac, Windows, or Linux) and using a code editor like **VS Code** or an IDE like **PyCharm**. We will explore this method in later chapters.
*   **Cloud-Based (Recommended for Beginners):** This is the easiest way to start. We will use **Google Colab**, which is a free, online tool that lets you write and execute Python code directly in your browser.
    *   **No installation required.**
    *   Access to powerful computing resources for free.
    *   Easy to share your work.

All lessons in this course are available as Google Colab notebooks.

## 3. Your First Python Program: The `print()` Function

The most basic task in any programming language is to display output. In Python, we use the built-in `print()` function.

```python
# This line of code will print the text "Hello, Data Nerd!" to the screen.
print("Hello, Data Nerd!")

# The Python interpreter executes code line by line, from top to bottom.
print("---")
print(" /|")
print("/_|")
```

The `print()` function is more powerful than it looks. You can even perform operations inside it. For example, you can repeat a string using the `*` operator:

```python
# This will print a line of 10 asterisks
print("*" * 10)
```

## 4. Variables and Primitive Data Types

**Variables** are fundamental to programming. Think of them as containers or labels for storing data in a computer's memory.

```python
# Assigning the integer 1000 to the variable 'student_count'
student_count = 1000
```

Python is a **dynamically typed language**, which means you don't have to specify the data type of a variable. The type is determined automatically at runtime.

The core (or "primitive") data types you'll use constantly are:

*   **Strings (`str`):** A sequence of characters, used for text. Defined with single (`'`) or double (`"`) quotes.
    ```python
    course_name = "Python for Data Science"
    ```
*   **Integers (`int`):** Whole numbers without a decimal point.
    ```python
    student_count = 1000
    ```
*   **Floats (`float`):** Numbers with a decimal point.
    ```python
    rating = 4.99
    ```
*   **Booleans (`bool`):** Represent truth values. They can only be `True` or `False` (with a capital T and F).
    ```python
    is_published = True
    ```

## 5. Getting User Input

You can make your programs interactive by getting input from the user with the built-in `input()` function.

**Important:** The `input()` function **always returns data as a string**.

```python
name = input("What is your name? ")
print("Hello, " + name)
```

If you expect a number, you must convert the string to the appropriate numeric type using functions like `int()` or `float()`.

```python
birth_year_str = input("Enter your birth year: ")
birth_year_int = int(birth_year_str)
age = 2025 - birth_year_int
print("You are", age, "years old.")
```

## 6. Working with Strings

Since text is such a common data type, Python provides many powerful ways to manipulate strings.

### Formatted Strings (f-strings)

F-strings are the modern and most readable way to embed expressions inside of strings. Prefix the string with an `f` and place your variables or expressions inside curly braces `{}`.

```python
first = "John"
last = "Smith"

# Old way (concatenation)
full_old = first + " " + last

# New way (f-string)
full_new = f"{first} {last}"

print(full_new) # Output: John Smith
```

You can even put expressions inside the curly braces:
```python
print(f"The length of the first name is: {len(first)}")
```

### Useful String Methods

Methods are functions that belong to a specific object. You call them using dot notation (e.g., `my_string.upper()`).

*   `upper()`: Converts the string to uppercase.
*   `lower()`: Converts the string to lowercase.
*   `title()`: Capitalizes the first letter of each word.
*   `find()`: Returns the index (starting from 0) of the first occurrence of a character or sequence of characters. It's case-sensitive.
*   `replace()`: Replaces a character or sequence of characters with another.

```python
course = "Python for Beginners"
print(course.upper()) # Output: PYTHON FOR BEGINNERS
print(course.find("for")) # Output: 7
print(course.replace("Beginners", "Absolute Beginners"))
```

## 7. Working with Numbers

Python supports all standard arithmetic operations.

*   `+` (Addition)
*   `-` (Subtraction)
*   `*` (Multiplication)
*   `/` (Float Division - always results in a float, e.g., `10 / 3` is `3.33...`)
*   `//` (Integer Division - results in an integer, e.g., `10 // 3` is `3`)
*   `%` (Modulus - gives the remainder of a division, e.g., `10 % 3` is `1`)
*   `**` (Exponentiation - e.g., `10 ** 3` is `1000`)

**Operator Precedence** follows standard mathematical rules:
1.  Parentheses `()`
2.  Exponentiation `**`
3.  Multiplication `*` and Division `/`, `//`
4.  Addition `+` and Subtraction `-`

## 8. Project: Number Guessing Game

Let's combine these concepts to build a simple number guessing game. The computer will think of a secret number, and the user has to guess it.

This project demonstrates:
*   Importing external modules (`random`).
*   Using variables to store state (`secret_number`, `guess`).
*   Getting and converting user input (`input()`, `int()`).
*   Using a `while` loop to repeat an action.
*   Using conditional logic (`if`, `elif`, `else`) to provide feedback.

```python
import random

def guess_game(x):
    # The computer generates a random number between 1 and x
    secret_number = random.randint(1, x)
    
    guess = 0 # Initialize guess to a number that can't be the secret number
    
    # Keep looping as long as the guess is not correct
    while guess != secret_number:
        # Get the user's guess (and convert it to an integer)
        guess = int(input(f"Guess a number between 1 and {x}: "))
        
        if guess < secret_number:
            print("Sorry, guess again. Too low.")
        elif guess > secret_number:
            print("Sorry, guess again. Too high.")
            
    # The loop ends when the guess is correct
    print(f"Yay, congrats! You have guessed the number {secret_number} correctly!")

# Let's play the game!
guess_game(10)
```
