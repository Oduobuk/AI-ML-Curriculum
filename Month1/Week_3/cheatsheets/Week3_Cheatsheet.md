# Week 3 Cheatsheet: Python Data Structures

## Lists `[]`
An **ordered**, **mutable** (changeable) collection of items.

*   **Creation:** `my_list = ["apple", 10, True]`
*   **Accessing:**
    *   By index: `my_list[0]` -> `"apple"`
    *   Negative index: `my_list[-1]` -> `True`
    *   Slicing: `my_list[1:]` -> `[10, True]`
*   **Key Methods:**
    *   `.append(item)`: Adds an item to the end.
    *   `.extend(another_list)`: Adds all items from another list to the end.
    *   `.insert(index, item)`: Adds an item at a specific position.
    *   `.remove(item)`: Removes the first occurrence of an item.
    *   `.pop()`: Removes and returns the last item.
    *   `.sort()`: Sorts the list in place.
    *   `.reverse()`: Reverses the list in place.
    *   `.index(item)`: Returns the index of the first occurrence of an item.
    *   `.copy()`: Returns a shallow copy of the list.

---

## Tuples `()`
An **ordered**, **immutable** (unchangeable) collection of items.

*   **Creation:** `my_tuple = ("apple", 10, True)`
*   **Accessing:** Same as lists (index and slicing).
    *   `my_tuple[0]` -> `"apple"`
*   **Why use tuples?**
    *   For data that should not change (e.g., coordinates, RGB color values).
    *   Slightly more memory-efficient than lists.
    *   Can be used as keys in a dictionary (lists cannot).
*   **Unpacking:** A powerful feature for assigning tuple items to variables.
    ```python
    coordinates = (10, 20)
    x, y = coordinates # x is 10, y is 20
    ```

---

## Dictionaries `{}`
An **unordered** collection of **key-value pairs**. Mutable.

*   **Creation:**
    ```python
    my_dict = {
        "name": "John",
        "age": 30,
        "is_student": False
    }
    ```
*   **Accessing Values:**
    *   Using the key: `my_dict["name"]` -> `"John"` (will raise an error if the key doesn't exist).
    *   Using the `.get()` method (safer): `my_dict.get("age")` -> `30`.
    *   `.get()` with a default value: `my_dict.get("city", "Unknown")` -> `"Unknown"`.
*   **Modifying and Adding:**
    *   Add a new pair: `my_dict["city"] = "New York"`
    *   Update an existing pair: `my_dict["age"] = 31`
*   **Key Methods:**
    *   `.keys()`: Returns a view of all keys.
    *   `.values()`: Returns a view of all values.
    *   `.items()`: Returns a view of all key-value pairs (as tuples).

---

## Control Flow

*   **`if` / `elif` / `else`:** For making decisions.
    ```python
    if temperature > 30:
        print("It's hot.")
    elif temperature < 10:
        print("It's cold.")
    else:
        print("It's nice.")
    ```
*   **`while` Loop:** Repeats a block of code as long as a condition is true.
    ```python
    i = 0
    while i < 5:
        print(i)
        i += 1 # Don't forget to increment!
    ```
*   **`for` Loop (for iterating over collections):**
    ```python
    # Looping through a list
    for friend in ["Jim", "Karen", "Kevin"]:
        print(friend)

    # Looping through a dictionary's items
    for key, value in my_dict.items():
        print(f"{key}: {value}")
    ```
