# Week 2 Cheatsheet: Python Fundamentals

## Primitive Data Types

| Data Type | Name    | Description                  | Example                          |
| --------- | ------- | ---------------------------- | -------------------------------- |
| **String**  | `str`   | A sequence of characters.    | `"Hello"`, `'Python'`            |
| **Integer** | `int`   | A whole number.              | `100`, `-5`                      |
| **Float**   | `float` | A number with a decimal point. | `4.99`, `-0.5`                   |
| **Boolean** | `bool`  | A truth value.               | `True`, `False` (case-sensitive) |

---

## Variables and Input

*   **Variable Assignment:** Use the `=` operator to assign a value to a variable.
    ```python
    student_count = 1000
    course_name = "Data Science"
    ```
*   **Getting User Input:** The `input()` function gets input from the user and **always returns it as a string**.
    ```python
    name = input("What is your name? ")
    ```
*   **Type Conversion:** Convert a value from one type to another.
    ```python
    # Convert string input to an integer
    age_str = input("Enter your age: ")
    age_int = int(age_str)

    # Convert an integer to a string
    count_str = str(100)
    ```

---

## String Manipulation

*   **Concatenation:** Combining strings with the `+` operator.
    ```python
    full_name = "John" + " " + "Smith"
    ```
*   **Formatted Strings (f-strings):** The best way to embed expressions in strings.
    ```python
    first = "John"
    last = "Smith"
    full_name = f"{first} {last}" # -> "John Smith"
    message = f"Hello, your name has {len(full_name)} characters."
    ```
*   **Common String Methods:**
    *   `.upper()`: Convert to uppercase.
    *   `.lower()`: Convert to lowercase.
    *   `.title()`: Capitalize each word.
    *   `.strip()`: Remove whitespace from the beginning and end.
    *   `.find('char')`: Find the index of a character. Returns -1 if not found.
    *   `.replace('old', 'new')`: Replace a substring.

---

## Arithmetic Operators

| Operator | Name                 | Example         | Result |
| -------- | -------------------- | --------------- | ------ |
| `+`      | Addition             | `5 + 2`         | `7`    |
| `-`      | Subtraction          | `5 - 2`         | `3`    |
| `*`      | Multiplication       | `5 * 2`         | `10`   |
| `/`      | Float Division       | `5 / 2`         | `2.5`  |
| `//`     | Integer Division     | `5 // 2`        | `2`    |
| `%`      | Modulus (Remainder)  | `5 % 2`         | `1`    |
| `**`     | Exponentiation       | `5 ** 2`        | `25`   |

*   **Augmented Assignment:** A shorthand for modifying a variable.
    *   `x = x + 3` is the same as `x += 3`

---

## Comparison and Logical Operators

*   **Comparison:** `==` (Equal), `!=` (Not Equal), `>` (Greater Than), `<` (Less Than), `>=`, `<=`
*   **Logical:** `and` (both must be true), `or` (at least one must be true), `not` (inverts the truth value)
