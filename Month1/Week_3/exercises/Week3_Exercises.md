# Week 3 Exercises: Data Structures and Loops

## Objective

These exercises will give you hands-on practice with Python's core data structures (lists, dictionaries) and control flow (`for` and `while` loops).

---

### Exercise 1: List Manipulation

**Goal:** Practice common list operations.

**Instructions:**
1.  Create a list of numbers, e.g., `numbers = [5, 2, 8, 1, 9, 4]`.
2.  Write code to find the largest number in the list *without* using the built-in `max()` function. (Hint: Use a `for` loop and an `if` statement).
3.  Use a list method to add the number `10` to the end of the list.
4.  Use a list method to sort the list in ascending order.
5.  Print the final, sorted list.

---

### Exercise 2: Removing Duplicates

**Goal:** Write a program that removes duplicate items from a list.

**Instructions:**
1.  Start with a list that contains duplicate values, e.g., `items = [2, 2, 4, 6, 3, 4, 6, 1]`.
2.  Create an empty list called `uniques`.
3.  Loop through the `items` list. For each item, check if it's already in the `uniques` list. If it's not, add it to `uniques`.
4.  Print the `uniques` list.

**Example Output:**
```
[2, 4, 6, 3, 1]
```

---

### Exercise 3: Simple Dictionary Translator

**Goal:** Use a dictionary to create a program that can "translate" a few words.

**Instructions:**
1.  Create a dictionary where the keys are English words and the values are their (simplified) translations in another language (e.g., Spanish, French, or even just a "code" language like Pig Latin).
    ```python
    # Example
    translator = {
        "hello": "hola",
        "goodbye": "adios",
        "cat": "gato"
    }
    ```
2.  Prompt the user to enter a word in English.
3.  Use the dictionary to find the translation of the word.
4.  Print the translation. Use the `.get()` method to handle cases where the word is not in the dictionary.

**Example Interaction:**
```
Enter a word to translate: cat
Translation: gato

Enter a word to translate: dog
Translation: I don't have a translation for that word.
```

---

### Exercise 4: Building a 2D List (Grid)

**Goal:** Use nested loops to create a 2D list (a list of lists), which can represent a grid.

**Instructions:**
1.  Define a `rows` variable and a `cols` variable (e.g., `rows = 3`, `cols = 4`).
2.  Create an empty list called `grid`.
3.  Use a `for` loop to iterate from 0 up to `rows`.
4.  Inside that loop, create another empty list called `row`.
5.  Inside the inner loop, use another `for` loop to iterate from 0 up to `cols`. In this loop, append a value (e.g., `0`) to the `row` list.
6.  After the inner loop finishes, append the `row` list to the `grid` list.
7.  Finally, print the `grid`.

**Expected Output (for rows=3, cols=4):**
```
[
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0]
]
```
This is a fundamental concept for working with grid-based data, like images or game boards.
