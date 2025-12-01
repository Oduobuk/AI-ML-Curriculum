# Week 4 Cheatsheet: Basic Plotting with Matplotlib

`matplotlib.pyplot` is a collection of functions that make matplotlib work like MATLAB. It's the most common way to create plots. We typically import it with the alias `plt`.

```python
import matplotlib.pyplot as plt
```

---

## The Basic Plotting Workflow

Creating a simple plot follows a standard sequence of steps:

1.  **Prepare Data:** Put your data into Python lists or NumPy arrays.
2.  **Create Plot:** Call a plotting function (e.g., `plt.plot()`, `plt.bar()`) to create the plot.
3.  **Customize:** Add titles, labels, and other customizations.
4.  **Show/Save:** Display the plot on the screen or save it to a file.

---

## Common Plot Types

### Line Plot
Used to show trends over time or continuous data.

```python
x_values = [0, 1, 2, 3, 4]
y_values = [0, 1, 4, 9, 16]

plt.plot(x_values, y_values)
```

### Bar Chart
Used to compare quantities among different categories.

```python
categories = ['A', 'B', 'C']
values = [10, 25, 15]

plt.bar(categories, values)
```

### Scatter Plot
Used to show the relationship between two numerical variables.

```python
x_coords = [1, 2, 3, 4, 5]
y_coords = [5, 7, 6, 8, 7]

plt.scatter(x_coords, y_coords)
```

---

## Essential Customization Functions

These functions are called *after* creating the plot but *before* showing it.

*   **`plt.figure(figsize=(width, height))`**
    *   Optional: Creates a new figure and allows you to set its size in inches. Good practice to start with this.
    *   Example: `plt.figure(figsize=(10, 6))`

*   **`plt.title("My Awesome Plot")`**
    *   Adds a title to the top of the plot.

*   **`plt.xlabel("My X-axis Label")`**
    *   Adds a label to the X-axis.

*   **`plt.ylabel("My Y-axis Label")`**
    *   Adds a label to the Y-axis.

*   **`plt.legend()`**
    *   Displays a legend on the plot. You must add a `label` to each plot element for the legend to work.
    *   Example: `plt.plot(x, y, label="Model 1")`

*   **`plt.grid(True)`**
    *   Adds a grid to the background of the plot.

*   **`plt.tight_layout()`**
    *   Adjusts plot parameters for a tight layout, often fixing issues with labels being cut off.

---

## Showing and Saving

*   **`plt.show()`**
    *   Displays the plot in a new window. This should typically be the last line of your script.

*   **`plt.savefig("my_plot.png")`**
    *   Saves the current figure to a file. You can use different extensions like `.png`, `.jpg`, `.pdf`, etc.
    *   **Important:** Call this *before* `plt.show()`, as `plt.show()` clears the figure.

---

## Full Example

```python
import matplotlib.pyplot as plt

# 1. Prepare Data
model_names = ['Model A', 'Model B', 'Model C']
accuracies = [85, 92, 78]

# 2. Create Plot
plt.figure(figsize=(8, 5)) # Optional: set figure size
plt.bar(model_names, accuracies, color='skyblue')

# 3. Customize
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100) # Set Y-axis limits

# 4. Show/Save
plt.savefig('accuracy_chart.png')
plt.show()
```
