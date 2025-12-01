# Week 3: Representing Data with Data Structures

This week, we are focusing on the core data structures in Python: **Lists, Tuples, and Dictionaries**.

While we are not yet working with large, external dataset files (like CSVs), this is the week where we learn how to *store* and *organize* the data that comes from them.

Think of data structures as the in-memory containers for your data. When you load a dataset from a file, you will almost always load it into one of these structures.

## How Data Structures Relate to Datasets

Imagine a simple dataset of student information stored in a spreadsheet:

| Name    | Major     | GPA   |
|---------|-----------|-------|
| John    | History   | 3.5   |
| Jane    | Biology   | 3.8   |
| Mike    | CS        | 3.2   |

Here are a few ways you could represent this data in Python using the structures you learned this week:

### 1. Using a List of Lists

Each inner list represents a row from the table.

```python
student_data = [
    ["John", "History", 3.5],
    ["Jane", "Biology", 3.8],
    ["Mike", "CS", 3.2]
]

# To get Jane's GPA:
janes_gpa = student_data[1][2]
print(janes_gpa) # Output: 3.8
```
*   **Pro:** Simple to construct.
*   **Con:** You have to remember the index for each piece of data (e.g., index `2` is GPA). This is not very readable.

### 2. Using a List of Dictionaries (Very Common!)

This is a very popular and readable way to structure data. Each dictionary in the list represents a single record (a row), and the keys tell you exactly what each piece of data means.

```python
student_data = [
    {"name": "John", "major": "History", "gpa": 3.5},
    {"name": "Jane", "major": "Biology", "gpa": 3.8},
    {"name": "Mike", "major": "CS", "gpa": 3.2}
]

# To get Jane's GPA:
janes_gpa = student_data[1]["gpa"]
print(janes_gpa) # Output: 3.8
```
*   **Pro:** Very clear and readable. No need to remember index positions.
*   **Con:** Slightly more verbose to type out.

The exercises and assignments for this week will focus on creating and manipulating these structures manually. In the coming weeks, you will learn to use libraries like **Pandas** that are designed to do this automatically from files.
