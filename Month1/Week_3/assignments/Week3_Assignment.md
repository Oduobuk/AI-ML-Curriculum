# Week 3 Assignment: Advanced Guessing Game

## Objective

This assignment will combine your knowledge of `while` loops, `if` statements, user input, and variables to create a word guessing game with a limited number of attempts. This is an extension of the simple game shown in the lecture notes.

## Instructions

1.  **Create a new Python file:** Name it `guessing_game.py`.

2.  **Set up the Game Variables:**
    *   Create a variable `secret_word` and assign it a string value (e.g., "python"). This is the word the user needs to guess.
    *   Create a variable `guess` and initialize it to an empty string `""`.
    *   Create a `guess_count` variable and initialize it to `0`.
    *   Create a `guess_limit` variable and set it to an integer (e.g., `3`). This is the maximum number of guesses the user gets.
    *   Create a boolean variable `out_of_guesses` and initialize it to `False`.

3.  **Implement the Game Logic with a `while` Loop:**
    *   Create a `while` loop that continues as long as the user's `guess` is not equal to the `secret_word` AND they are not `out_of_guesses`.
    *   Inside the loop, use an `if` statement to check if the `guess_count` is less than the `guess_limit`.
        *   If it is, prompt the user to enter their guess and increment the `guess_count`.
        *   If it's not, set `out_of_guesses` to `True` to end the loop.

4.  **Determine the Winner or Loser:**
    *   After the loop finishes, use an `if` statement to check the value of `out_of_guesses`.
    *   If `out_of_guesses` is `True`, print a "You lose!" message.
    *   If it's `False` (meaning the user guessed the word correctly), print a "You win!" message.

## Example Interaction

**Scenario 1: User wins**
```
Enter guess: java
Enter guess: c++
Enter guess: python
You win!
```

**Scenario 2: User loses**
```
Enter guess: java
Enter guess: c++
Enter guess: ruby
Out of Guesses, YOU LOSE!
```

## Bonus Challenge (Optional)

*   **Give Hints:** After a wrong guess, can you give the user a hint? For example, tell them how many letters are in the secret word.
*   **Play Again:** After the game ends, ask the user if they want to play again. If they say "yes", reset the game variables and start over.

## Submission

*   Save your completed Python code in the `guessing_game.py` file.
*   Be prepared to demonstrate your working game.
