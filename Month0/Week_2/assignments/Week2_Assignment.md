Week 2: Assignment - Student Management System

📋 Project Overview

Build a Student Management System with functions for adding, updating, and analyzing student data.

Skills Tested:
- Functions with parameters and return values
- Lists and dictionaries
- Loops and conditionals
- Data manipulation

🎯 Requirements

Part 1: Data Structure (20%)
Create a dictionary that stores students with their information:
```python
students = {
    "S001": {"name": "Alice", "grade": 85, "major": "CS"},
    "S002": {"name": "Bob", "grade": 92, "major": "Math"},
    # ...
}
```

Part 2: Functions (60%)
Implement the following functions:

1. `addstudent(studentid, name, grade, major)`
- Adds a new student to the system
- Returns True if successful, False if student already exists

2. `getstudent(studentid)`
- Returns student data or None if not found

3. `updategrade(studentid, new_grade)`
- Updates a student's grade
- Returns True if successful, False if student not found

4. `listallstudents()`
- Prints all students in a formatted table
- Shows: ID, Name, Grade, Major

5. `calculateaveragegrade()`
- Returns the average grade of all students

6. `gettopperformer()`
- Returns the name of the student with the highest grade

7. `getstudentsby_major(major)`
- Returns list of students in a specific major

8. `deletestudent(studentid)`
- Removes a student from the system
- Returns True if successful, False if student not found

Part 3: Interactive Menu (20%)
Create a main program with a menu:
```
========== STUDENT MANAGEMENT SYSTEM ==========
1. Add Student
2. View Student
3. Update Grade
4. List All Students
5. Calculate Average Grade
6. Find Top Performer
7. Find Students by Major
8. Delete Student
9. Exit

Enter your choice (1-9):
```

🚀 Implementation Guide

Step 1: Set Up Data Structure
```python
students = {}  # Empty dictionary to start
```

Step 2: Implement Functions
Test each function individually before moving on.

Step 3: Create Main Menu Loop
```python
def main():
    while True:
        # Print menu
        # Get user choice
        # Call appropriate function
        # Display results
```

💡 Tips

- Use meaningful variable names
- Add comments explaining complex logic
- Handle edge cases (empty lists, invalid input)
- Format output nicely for readability

✅ Checklist

- [ ] All 8 functions implemented correctly
- [ ] Menu system works and handles invalid input
- [ ] Can add, view, update, and delete students
- [ ] Calculations (average, top performer) are correct
- [ ] Output is formatted nicely
- [ ] Code is well-commented
- [ ] No crashes on edge cases

🎯 Sample Output

```
========== STUDENT MANAGEMENT SYSTEM ==========
1. Add Student
2. View Student
3. Update Grade
4. List All Students
5. Calculate Average Grade
6. Find Top Performer
7. Find Students by Major
8. Delete Student
9. Exit

Enter your choice: 1
Enter Student ID: S001
Enter Name: Alice
Enter Grade: 85
Enter Major: Computer Science
✓ Student added successfully!

Enter your choice: 4
--- ALL STUDENTS ---
ID    | Name    | Grade | Major
------|---------|-------|-------------------
S001  | Alice   | 85    | Computer Science
S002  | Bob     | 92    | Mathematics

Enter your choice: 5
Average Grade: 88.5

Enter your choice: 6
Top Performer: Bob (Grade: 92)
```

📤 Submission

1. Save as `studentmanagementsystem.py`
2. Test thoroughly with multiple inputs
3. Include sample run-through in comments

[Back to Lecture Notes](../lecturenotes/Week2ControlFlowFunctions.md)
