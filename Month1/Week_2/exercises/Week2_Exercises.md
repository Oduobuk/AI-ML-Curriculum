# Week 2 Exercises: Python Fundamentals Practice

## Objective

These exercises are designed to give you hands-on practice with the fundamental Python concepts covered in Week 2. Each exercise should be written in its own Python file (e.g., `exercise1.py`).

---

### Exercise 1: Mad Libs Game

**Goal:** Create a simple "Mad Libs" game where the user is prompted for different words, and then a story is generated using their input.

**Instructions:**
1.  Prompt the user to enter several words: a color, a plural noun, and a celebrity name.
2.  Store each of the user's inputs in its own variable.
3.  Use an f-string to combine the user's input into a short, funny story.
4.  Print the final story to the screen.

**Example Interaction:**
```
Enter a color: blue
Enter a plural noun: potatoes
Enter a celebrity: Tom Hanks

Roses are blue,
potatoes are violet,
I love Tom Hanks.
```

---

### Exercise 2: Simple Age Calculator

**Goal:** Write a program that asks the user for their birth year and calculates their approximate age.

**Instructions:**
1.  Prompt the user to enter their birth year using the `input()` function.
2.  Convert the user's input from a string to an integer.
3.  Calculate the user's age by subtracting their birth year from the current year (you can hardcode the current year, e.g., 2025).
4.  Print the calculated age in a user-friendly sentence.

**Example Interaction:**
```
Enter your birth year: 1995
You are 30 years old.
```

---

### Exercise 3: Temperature Converter

**Goal:** Build a program that converts a temperature from Fahrenheit to Celsius.

**Instructions:**
1.  Prompt the user to enter a temperature in Fahrenheit.
2.  Convert the input string to a number (a `float` would be best for this).
3.  Use the formula `Celsius = (Fahrenheit - 32) * 5/9` to perform the conversion.
4.  Print the result in a clear format, showing both the original Fahrenheit temperature and the converted Celsius temperature.

**Example Interaction:**
```
Enter temperature in Fahrenheit: 68
68.0°F is equal to 20.0°C.
```

---

### Exercise 4: String Analysis

**Goal:** Write a program that takes a sentence from the user and provides some basic analysis.

**Instructions:**
1.  Prompt the user to enter a sentence.
2.  Calculate and print the total number of characters in the sentence (hint: use the `len()` function).
3.  Convert the entire sentence to uppercase and print the result.
4.  Ask the user for a specific word to find within their sentence. Use the `.find()` method to see if the word exists and print its starting index.

**Example Interaction:**
```
Please enter a sentence: The quick brown fox jumps over the lazy dog.
Your sentence has 44 characters.
IN ALL CAPS: THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.

What word should I find? fox
The word 'fox' starts at index 16.
```
