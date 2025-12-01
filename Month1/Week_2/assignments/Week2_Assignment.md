# Week 2 Assignment: Simple Command-Line Calculator

## Objective

The goal of this assignment is to apply your knowledge of Python fundamentals, including user input, data type conversion, and conditional logic (`if` statements), to build a simple calculator that works in the command line.

## Instructions

1.  **Create a new Python file:** Name it `calculator.py`.

2.  **Prompt the User for Input:** Your program should ask the user for three pieces of information:
    *   The first number.
    *   The operator (e.g., `+`, `-`, `*`, `/`).
    *   The second number.

3.  **Perform the Calculation:**
    *   Based on the operator provided by the user, your program should perform the correct mathematical operation.
    *   You will need to use `if`, `elif`, and `else` statements to check which operator was entered.

4.  **Print the Result:**
    *   Display the result of the calculation to the user in a clear and readable format. For example: `10 + 5 = 15`.

5.  **Handle Data Types:**
    *   Remember that the `input()` function returns strings. You will need to convert the numeric input from the user into numbers (e.g., `float`) before you can perform calculations.

## Example Interaction

Here is how your program should behave when it's run:

```
Enter the first number: 10
Enter an operator (+, -, *, /): +
Enter the second number: 5
10.0 + 5.0 = 15.0
```

Another example:
```
Enter the first number: 100
Enter an operator (+, -, *, /): /
Enter the second number: 4
100.0 / 4.0 = 25.0
```

## Bonus Challenge (Optional)

*   **Error Handling:** What happens if the user tries to divide by zero? Add a check to prevent this and print an error message like "Error: Cannot divide by zero."
*   **Invalid Operator:** What if the user enters an operator that isn't `+`, `-`, `*`, or `/`? Add a check for this and print an error message.

## Submission

*   Save your completed Python code in the `calculator.py` file.
*   Be prepared to demonstrate your working calculator.
