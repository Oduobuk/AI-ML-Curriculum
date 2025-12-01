# Week 4 Exercises: Matplotlib Practice

## Objective

These exercises are designed to give you basic, hands-on practice creating common plots using the `matplotlib` library.

---

### Exercise 1: Simple Line Plot

**Goal:** Create a line plot to visualize the growth of a hypothetical investment over time.

**Instructions:**
1.  Create a new Python file, `line_plot_exercise.py`.
2.  Import `matplotlib.pyplot` as `plt`.
3.  Create two lists:
    *   `years`: `[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]`
    *   `investment_value`: `[1000, 1150, 1300, 1400, 1600, 1550, 1700, 1850, 2000, 2200]`
4.  Create a line plot with `years` on the X-axis and `investment_value` on the Y-axis.
5.  Add a title: "Investment Growth Over Time".
6.  Add an X-axis label: "Year".
7.  Add a Y-axis label: "Value (USD)".
8.  Display the plot using `plt.show()`.

---

### Exercise 2: Student Grades Bar Chart

**Goal:** Create a bar chart to compare the final grades of several students.

**Instructions:**
1.  Create a new Python file, `bar_chart_exercise.py`.
2.  Import `matplotlib.pyplot` as `plt`.
3.  Create two lists:
    *   `students`: `["Alice", "Bob", "Charlie", "David", "Eve"]`
    *   `grades`: `[88, 92, 75, 98, 85]`
4.  Create a bar chart with `students` on the X-axis and `grades` on the Y-axis.
5.  Add a title: "Final Grades for Python 101".
6.  Add an X-axis label: "Student".
7.  Add a Y-axis label: "Final Score".
8.  Set the Y-axis limits to be from 0 to 100 using `plt.ylim(0, 100)`. This provides better context for grades.
9.  Display the plot using `plt.show()`.

---

### Exercise 3: Multiple Lines on the Same Plot

**Goal:** Compare the monthly sales of two different products on the same line plot.

**Instructions:**
1.  Create a new Python file, `multiple_lines_exercise.py`.
2.  Import `matplotlib.pyplot` as `plt`.
3.  Create the following lists:
    *   `months`: `["Jan", "Feb", "Mar", "Apr", "May", "Jun"]`
    *   `product_a_sales`: `[120, 135, 140, 130, 150, 160]`
    *   `product_b_sales`: `[80, 85, 90, 105, 100, 110]`
4.  Plot the sales for Product A. Use the `label` argument in your `plt.plot()` call (e.g., `label="Product A"`). You can also add a `marker='o'` to show data points.
5.  On the same plot, plot the sales for Product B. Also give it a label (e.g., `label="Product B"`).
6.  Add a title: "Monthly Sales Comparison".
7.  Add an X-axis label: "Month".
8.  Add a Y-axis label: "Units Sold".
9.  Call `plt.legend()` to display the legend.
10. Display the plot using `plt.show()`.
