Week 1: Exercises

Getting Comfortable with Python

Complete these exercises to reinforce the concepts from the lecture. Start with Easy, then Medium, then Challenge.

🟢 Easy Exercises

Exercise 1.1: Hello World Variations

Create a program that prints the following (exactly):
```
*
    Welcome to Python!
*
```

Hint: Use `print()` and string multiplication.

Exercise 1.2: Variables & Types

Create a program that:
1. Creates variables for: name, age, height (float), is_student (boolean)
2. Prints each variable with a label
3. Prints the type of each variable

Example Output:
```
Name: Alice
Age: 25
Height: 5.7
Is Student: True
Type of name: <class 'str'>
Type of age: <class 'int'>
```

Exercise 1.3: Simple Arithmetic

Write a program that:
1. Stores two numbers in variables
2. Calculates: sum, difference, product, quotient, power
3. Prints all results with labels

Example:
```
Number 1: 10
Number 2: 3

10 + 3 = 13
10 - 3 = 7
10 * 3 = 30
10 / 3 = 3.333...
10  3 = 1000
```

Exercise 1.4: String Practice

Create a program that:
1. Stores a first name and last name
2. Combines them into a full name
3. Prints the full name
4. Prints the length of the full name

Example:
```
First Name: John
Last Name: Doe
Full Name: John Doe
Length: 8
```

Exercise 1.5: Input & Output

Write a program that:
1. Asks the user for their name
2. Asks the user for their favorite color
3. Prints: "Hi [name], I like [color] too!"

🟡 Medium Exercises

Exercise 2.1: Rectangle Area

Create a program that:
1. Gets width and height from user (as floats)
2. Calculates the area
3. Prints results

Example:
```
Enter width: 5
Enter height: 3
Area: 15.0
```

Exercise 2.2: Temperature Converter

Write a program that:
1. Gets temperature in Celsius from user
2. Converts to Fahrenheit: F = (C × 9/5) + 32
3. Prints both temperatures

Example:
```
Enter temperature in Celsius: 0
0.0°C = 32.0°F
```

Exercise 2.3: Age Calculator

Create a program that:
1. Gets the user's birth year
2. Calculates current age (assume 2024)
3. Calculates age 10 years from now
4. Prints both

Exercise 2.4: Billing Calculator

Write a program that:
1. Gets item price from user
2. Calculates 8% tax
3. Calculates total (price + tax)
4. Prints all three values

Example:
```
Item Price: $50.00
Tax (8%): $4.00
Total: $54.00
```

Hint: Use f-strings for nice formatting with `:.2f` for 2 decimal places.

Exercise 2.5: Personal Summary

Create a program that:
1. Gets user's name, age, city
2. Gets their favorite hobby
3. Prints a formatted summary

Example Output:
```
=== PERSONAL PROFILE ===
Name: Alice
Age: 25
City: New York
Hobby: Reading
Favorite Activity: Alice loves Reading in New York!
```

🔴 Challenge Exercises

Challenge 2.1: Simple Tip Calculator

Create a program that:
1. Gets meal cost from user
2. Gets tip percentage from user (e.g., 15, 18, 20)
3. Calculates tip amount and total
4. Displays results nicely

Example:
```
Meal Cost: $75.50
Tip Percentage: 18
Tip Amount: $13.59
Total: $89.09
```

Challenge 2.2: BMI Calculator

Create a program that:
1. Gets weight (in pounds) from user
2. Gets height (in inches) from user
3. Calculates BMI: BMI = (weight / (height²)) × 703
4. Prints the BMI

Example:
```
Weight (lbs): 180
Height (inches): 70
BMI: 25.81
```

Challenge 2.3: Multiple Unit Conversion

Create a program that converts:
1. Miles to Kilometers (multiply by 1.60934)
2. Pounds to Kilograms (divide by 2.20462)
3. Fahrenheit to Celsius (C = (F - 32) × 5/9)

Let the user pick which conversion and perform it.

Challenge 2.4: Grocery Receipt

Create a program that:
1. Gets 3 item names and prices from user
2. Calculates subtotal
3. Calculates 6% tax
4. Calculates total
5. Prints a formatted receipt

Example Output:
```
===== GROCERY RECEIPT =====
Item 1: Bread         $2.50
Item 2: Milk          $3.75
Item 3: Eggs          $4.25

Subtotal:             $10.50
Tax (6%):             $0.63
Total:                $11.13
============================
```

🧠 Debugging Practice

These programs have bugs. Find and fix them:

Bug Fix 1:
```python
name = input("What's your name? ")
age_str = input("How old are you? ")
age = age_str  # Bug here!
print(f"{name} is {age + 1} next year")
```

Problem: What goes wrong? How do you fix it?

Bug Fix 2:
```python
price = 100
tax_rate = 0.08
total = price + (price tax_rate)  # Bug here!
print(total)
```

Problem: What's the syntax error?

Bug Fix 3:
```python
first = "John"
last = "Doe"
full_name = first last  # Bug here!
print(full_name)
```

Problem: What operator is missing?

💡 Tips for Success

1. Run frequently: Test after every 2-3 lines
2. Read error messages: They tell you what's wrong
3. Use meaningful names: `student_name` not `x`
4. Add comments: Explain your code
5. Don't memorize: Understanding is more important

🎯 Self-Check

After completing these exercises, you should be able to:
- [ ] Write programs using `print()` and `input()`
- [ ] Create and manipulate variables
- [ ] Perform arithmetic operations
- [ ] Work with strings (concatenation, length)
- [ ] Convert between data types
- [ ] Read and understand error messages

📝 Submission

Save your programs as `.py` files:
- `exercise11.py`
- `exercise12.py`
- etc.

You can organize them in a folder: `Week1_Exercises/`

Next: Move to the [Assignment](../assignments/Week1_Assignment.md)
