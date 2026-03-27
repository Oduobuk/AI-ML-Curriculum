Week 1: Assignment - Python Fundamentals Project

📋 Overview

In this assignment, you'll build a Personal Information System that collects user data, performs calculations, and displays a formatted report.

Skills Tested:
- Variables and data types
- User input/output
- Arithmetic operations
- String operations and formatting
- Type conversion

🎯 Project Requirements

Create a Python program called `personalinfosystem.py` that does the following:

Part 1: Data Collection (30%)
Collect the following information from the user:
1. Full name (string)
2. Age (integer)
3. Height in inches (float)
4. Weight in pounds (float)
5. Favorite hobby (string)

Part 2: Calculations (40%)
Calculate the following:
1. Years until retirement: Assume retirement age is 65
2. BMI (Body Mass Index): BMI = (weight / (height²)) × 703
3. Age in 10 years
4. Year born (approximate, using 2024)

Part 3: Formatted Output (30%)
Display a nicely formatted report that includes:
- A decorative header
- All original user information
- All calculated values
- A closing statement
- Proper spacing and alignment

📝 Example Output

```
╔═══════════════════════════════════════════════════════════════╗
║              PERSONAL INFORMATION SYSTEM REPORT               ║
╚═══════════════════════════════════════════════════════════════╝

--- PERSONAL INFORMATION ---
Name:                          John Doe
Age:                           30 years old
Height:                        70 inches
Weight:                        180 pounds
Favorite Hobby:                Playing Guitar

--- STATISTICS & PROJECTIONS ---
Approximate Birth Year:        1994
Age in 10 Years:               40 years old
Years Until Retirement (65):   35 years

--- HEALTH METRICS ---
BMI:                           25.81
BMI Category:                  Overweight (25 or higher)

--- END REPORT ---
Thank you for using the Personal Information System!
```

🚀 Step-by-Step Guide

Step 1: Get User Input
```python
Ask for and store each piece of information
name = input("Enter your full name: ")
... continue for other fields
```

Step 2: Perform Calculations
```python
Calculate years until retirement
yearstoretirement = 65 - age
... continue for other calculations
```

Step 3: Create the Report
```python
Print formatted output
print("╔═══════════════════════════════════════════╗")
print("║      PERSONAL INFORMATION SYSTEM         ║")
print("╚═══════════════════════════════════════════╝")
... continue with formatted output
```

💡 Implementation Tips

Formatting Tips
- Use f-strings for variable insertion: `print(f"Name: {name}")`
- Use `:>30` for right alignment: `f"{'Age:':<30} {age}"`
- Use escape characters: `\n` for newline, `\t` for tab

Calculation Tips
```python
BMI calculation
bmi = (weight / (height  2)) * 703

Years to retirement
yearstoretirement = 65 - age

Birth year (approximate)
birth_year = 2024 - age
```

Debugging Tips
- Test one section at a time
- Print intermediate values to verify calculations
- Check for type errors (string vs number)

✅ Checklist Before Submission

- [ ] Program runs without errors
- [ ] Collects all 5 pieces of user information
- [ ] Calculates all 4 required values correctly
- [ ] Output is nicely formatted and readable
- [ ] Used f-strings for formatting
- [ ] Code has comments explaining sections
- [ ] Variable names are clear and descriptive
- [ ] File is named `personalinfosystem.py`

🔧 Enhancement Ideas (Optional)

If you finish early, try adding:
1. Additional calculations: Age in months, days
2. BMI category: Display "Underweight", "Normal", "Overweight", etc.
3. Multiple profiles: Loop to let user enter info for multiple people
4. Data validation: Check that age is reasonable, height is positive, etc.

📤 Submission

1. Save your file as: `personalinfosystem.py`
2. Create a folder: `Week1_Assignment`
3. Move your file into it
4. Test your program one more time with sample input
5. Document your work and submit

🎓 Learning Outcomes

By completing this assignment, you'll have practiced:
- ✅ Collecting user input
- ✅ Type conversion (string to int/float)
- ✅ Arithmetic operations
- ✅ String formatting and display
- ✅ Writing programs with multiple steps
- ✅ Organizing code logically

📞 If You Get Stuck

Common Issues:

| Problem | Solution |
|---------|----------|
| "ValueError: could not convert string to float" | Use `float()` instead of `int()` for decimals |
| Output looks messy/unaligned | Use f-string alignment: `f"{name:<30}"` |
| BMI calculation seems wrong | Remember: height² means height  height, not just height  2 |
| Name has spaces and doesn't work | `input()` works fine with spaces; that's not the issue |

Where to Get Help:
- Review the lecture notes
- Check similar examples in exercises
- Google the error message
- Ask in the course forum

🌟 Sample Solution Structure

```python
Personal Information System

Step 1: Collect user information
name = input("Enter your full name: ")
age = int(input("Enter your age: "))
... continue

Step 2: Perform calculations
yearstoretirement = 65 - age
bmi = (weight / (height  2)) * 703
... continue

Step 3: Display formatted report
print("╔═══════════════════════════════════════════╗")
print("║      PERSONAL INFORMATION SYSTEM         ║")
print("╚═══════════════════════════════════════════╝")
... continue
```

Good luck! Remember: the goal is to practice, not to be perfect. Every line of code you write builds your skills. 💪

After Submission: Move on to [Week 2: Control Flow & Functions](../../Week2/lecturenotes/)
