
# Week 3: Core Python Data Structures and Control Flow

## 1. Introduction to Data Structures

In the real world, data rarely exists as single, isolated values. More often, we work with collections of related data. Data structures are containers that allow us to store and organize these collections in an efficient way. This week, we will explore Python's three fundamental data structures: Lists, Tuples, and Dictionaries.

## 2. Python Lists

A list is an ordered, mutable (changeable) collection of items. You can think of it as a versatile, dynamic array.

### Creating and Accessing Lists

You create a list by placing items inside square brackets `[]`, separated by commas.

```python
# A list can contain items of different data types
friends = ["Kevin", "Karen", "Jim"]
numbers = [1, 2, 3, 4, 5]
mixed = ["Oscar", 50, True]

# Accessing items is done via indexing (starts at 0)
print(friends[0])  # Output: Kevin
print(friends[-1]) # Output: Jim (negative indexing starts from the end)

# Slicing works just like with strings
print(friends[1:]) # Output: ['Karen', 'Jim']
```

### Key List Methods

Because lists are mutable, there are many methods to modify them:

*   `.extend(list2)`: Appends all items from another list to the end.
    ```python
    lucky_numbers = [4, 8, 15, 16, 23, 42]
    friends.extend(lucky_numbers)
    print(friends) # friends list is now longer
    ```
*   `.append(item)`: Adds a single item to the end of the list.
    ```python
    friends.append("Creed")
    ```
*   `.insert(index, item)`: Adds an item at a specified position.
    ```python
    friends.insert(1, "Kelly") # Adds "Kelly" at index 1
    ```
*   `.remove(item)`: Removes the first occurrence of a specified item.
    ```python
    friends.remove("Jim")
    ```
*   `.pop()`: Removes and returns the item at the end of the list.
*   `.sort()`: Sorts the list in ascending order.
*   `.reverse()`: Reverses the order of the list.
*   `.copy()`: Returns a copy of the list.

## 3. Python Tuples

A tuple is an ordered, **immutable** (unchangeable) collection of items. The key difference from a list is that once a tuple is created, you cannot change its contents.

### Creating and Using Tuples

You create a tuple with parentheses `()`.

```python
# Tuples are often used for data that should not change, like coordinates
coordinates = (4, 5)

print(coordinates[0]) # Output: 4
# coordinates[1] = 10 # This would cause an error! Tuples are immutable.
```

### Unpacking

A powerful feature of tuples (and lists) is unpacking, which allows you to assign the items of the collection to multiple variables at once.

```python
x, y = coordinates
print(x) # Output: 4
print(y) # Output: 5
```

## 4. Python Dictionaries

A dictionary is an unordered collection of **key-value pairs**. Instead of being indexed by numbers, dictionaries are indexed by unique keys. They are perfect for storing related pieces of information.

### Creating and Accessing Dictionaries

You create a dictionary with curly braces `{}`, with each item being a `key: value` pair.

```python
# Using month names as keys to store their full names
month_conversions = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
}

# Access the value by its key
print(month_conversions["Mar"]) # Output: March

# A safer way is using the .get() method, which can provide a default value
print(month_conversions.get("Dec", "Not a valid key")) # Output: Not a valid key
```

## 5. Control Flow: `if` Statements

Control flow allows your program to make decisions. `if` statements execute a block of code only if a certain condition is true.

### Structure and Operators

The structure is `if`, `elif` (else if), and `else`.

*   **Comparison Operators:** Used to compare values (`==`, `!=`, `>`, `<`, `>=`, `<=`).
*   **Logical Operators:** Used to combine conditions (`and`, `or`, `not`).

```python
is_hot = True
is_cold = False

if is_hot:
    print("It's a hot day. Drink plenty of water.")
elif is_cold:
    print("It's a cold day. Wear warm clothes.")
else:
    print("It's a lovely day.")
```

## 6. Control Flow: `while` Loops

`while` loops allow you to execute a block of code repeatedly as long as a certain condition remains true.

### Structure and Example

The loop continues until the condition evaluates to `False`.

```python
# A simple loop that prints numbers from 1 to 5
i = 1
while i <= 5:
    print(i)
    i = i + 1 # Increment the counter to eventually end the loop

print("Loop finished.")
```

### Example: A Simple Guessing Game

`while` loops are perfect for games where you need to repeat an action until a goal is met.

```python
secret_word = "giraffe"
guess = ""
guess_count = 0
guess_limit = 3
out_of_guesses = False

while guess != secret_word and not(out_of_guesses):
    if guess_count < guess_limit:
        guess = input("Enter guess: ")
        guess_count += 1
    else:
        out_of_guesses = True

if out_of_guesses:
    print("Out of Guesses, YOU LOSE!")
else:
    print("You win!")
```
This example demonstrates how variables, user input, `if` statements, and `while` loops all work together to create a functional program.
