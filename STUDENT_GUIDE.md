# Student Guide: How to Submit Your Work

This guide explains the standard "Fork and Pull Request" workflow we will use for all submissions. Following these steps is crucial for tracking your progress and receiving feedback.

## The Workflow Overview

You will not push your work directly to the main course repository. Instead, you will:
1.  Create your own copy of the repository on GitHub (a "Fork").
2.  Do your work on a separate "branch" within your copy.
3.  "Request" to have your work added to the main repository's `submissions` branch by opening a "Pull Request".

---

## Step-by-Step Instructions

### Step 1: Fork the Repository (Do this only once)

1.  Navigate to the main course repository on GitHub: `https://github.com/Oduobuk/AI-ML-Curriculum`
2.  In the top-right corner of the page, click the **"Fork"** button.
3.  This will create a complete copy of the repository under your own GitHub account.

### Step 2: Clone Your Fork Locally (Do this only once)

1.  On your GitHub account, navigate to your newly forked repository (e.g., `https://github.com/<YourUsername>/AI-ML-Curriculum`).
2.  Click the green **"<> Code"** button.
3.  Copy the HTTPS or SSH URL.
4.  Open a terminal on your local machine and run:
    ```bash
    git clone <the_url_you_copied>
    ```
5.  Navigate into the cloned directory:
    ```bash
    cd AI-ML-Curriculum
    ```

### Step 3: Create a New Branch for Your Weekly Work (Do this for every submission)

Before starting a new assignment, make sure your fork is up-to-date with the main course repository and create a new branch for your work.

1.  **Keep your main branch clean and updated.**
    ```bash
    # Switch to your main branch
    git checkout main

    # Add the original course repo as a remote source named "upstream" (you only have to do this once)
    git remote add upstream https://github.com/Oduobuk/AI-ML-Curriculum.git

    # Fetch the latest changes from the main course repo
    git pull upstream main
    ```

2.  **Create a new branch for your assignment.** Use a descriptive name.
    ```bash
    # Example for Month 1, Week 2 assignment
    git checkout -b m1-w2-assignment-solution
    ```

### Step 4: Do Your Work

1.  Create a directory for your submissions if it doesn't exist. Use your name to keep things organized.
    ```
    AI-ML-Curriculum/
    └── submissions/
        └── Your_Name/
            ├── Month1_Week1/
            │   ├── assignment.ipynb
            │   └── exercises.py
            └── Month1_Week2/
                └── ...
    ```
2.  Add all your assignment files, exercises, and project files into the appropriate weekly folder.

### Step 5: Commit and Push Your Work

1.  Add your files to git's tracking:
    ```bash
    # Add all changes
    git add .
    ```
2.  Commit your changes with a clear message:
    ```bash
    git commit -m "Completed Month 1 Week 2 Assignment"
    ```
3.  Push your new branch to your forked repository on GitHub:
    ```bash
    # The -u flag sets the upstream for the branch so you can just use 'git push' next time
    git push -u origin m1-w2-assignment-solution
    ```

### Step 6: Open a Pull Request (The Submission)

1.  Go to your forked repository on GitHub (e.g., `https://github.com/<YourUsername>/AI-ML-Curriculum`).
2.  GitHub will usually show a yellow banner suggesting you create a Pull Request from your recently pushed branch. Click the **"Compare & pull request"** button.
3.  If you don't see the banner, go to the "Pull requests" tab and click "New pull request".
4.  **THIS IS THE MOST IMPORTANT STEP.** Configure the Pull Request correctly:
    -   **Base repository**: `Oduobuk/AI-ML-Curriculum`
    -   **Base branch**: `submissions`  <-- Make sure this is `submissions`, NOT `main`.
    -   **Head repository**: `<YourUsername>/AI-ML-Curriculum`
    -   **Head branch**: `m1-w2-assignment-solution` (or whatever you named your branch)
5.  Give your Pull Request a title (e.g., "Submission: M1-W2 Assignment by [Your Name]").
6.  Click **"Create pull request"**.

You have now successfully submitted your work. You can see any feedback or comments from the instructor directly on the Pull Request page.