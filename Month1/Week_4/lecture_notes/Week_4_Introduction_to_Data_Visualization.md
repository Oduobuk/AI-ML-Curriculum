
# Week 4: Introduction to Data Visualization

## 1. The Python Visualization Landscape

Python has a vast and sometimes confusing landscape of data visualization libraries. However, at the heart of this ecosystem is **Matplotlib**. It is the foundational library upon which many other higher-level tools like Seaborn and even the built-in plotting functions in Pandas are built.

While other libraries like Plotly and Bokeh offer more interactivity using JavaScript, learning Matplotlib is the essential first step. It gives you the power to create almost any plot imaginable and provides a deep understanding of how plotting works in Python.

## 2. Getting Started: Your First Plot

To begin, we need to import the `pyplot` module from Matplotlib, which contains the functions for creating plots. The standard convention is to import it with the alias `plt`.

```python
import matplotlib.pyplot as plt

# We can create simple lists of data to plot
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6, 8]

# Create a line plot
plt.plot(x, y)

# Use plt.show() to display the graph
plt.show()
```

## 3. Customizing Your Plot

A basic plot is good, but a great plot is customized with titles, labels, and colors to make it readable and informative.

### Titles and Labels

You can add a title and labels for the x and y-axes. You can also customize the font.

```python
# Re-plotting our x and y
plt.plot(x, y)

# Add a title
plt.title("Our First Graph!", fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})

# Add X and Y labels
plt.xlabel("X Axis!")
plt.ylabel("Y Axis!")

plt.show()
```

### Ticks

The markers on the axes are called "ticks". We can explicitly set their position and labels.

```python
plt.plot(x, y)
plt.title("Our First Graph!")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# Set the ticks for the x and y axes
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([0, 2, 4, 6, 8, 10])

plt.show()
```

### Legends

When you have multiple lines on a graph, a legend is essential to tell them apart. To create a legend, you must first add a `label` to each plot.

```python
plt.plot(x, y, label='2x')
plt.legend()
plt.show()
```

### Line Customization

You can customize the color, width, style, and markers of your line.

```python
# Use parameters to customize the line
plt.plot(x, y, label='2x', color='red', linewidth=2, marker='.', markersize=10, markeredgecolor='blue')
plt.legend()
plt.show()
```

There is also a shorthand notation to combine color, marker, and line style: `fmt = '[color][marker][line]'`.

```python
# Shorthand: plot a green line with dot markers and a dashed line style
plt.plot(x, y, 'g.--', label='2x')
plt.legend()
plt.show()
```

## 4. Plotting Multiple Lines

You can plot multiple lines on the same graph by calling the `plt.plot()` function for each line before calling `plt.show()`.

Let's use **NumPy**, a powerful library for numerical operations, to create more complex data for a second line.

```python
import numpy as np

# Plot our first line
plt.plot(x, y, 'r.--', label='2x')

# Create data for a second line: y = x^2
# Use np.arange to create a more detailed x-axis
x2 = np.arange(0, 4.5, 0.5)
plt.plot(x2, x2**2, 'b-', label='x^2') # Plot x^2 with a solid blue line

plt.title("Our First Graph!")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.xticks(np.arange(0, 5, 1)) # Set ticks from 0 to 4
plt.legend()
plt.show()
```

## 5. Bar Charts

Bar charts are used to compare values across a set of discrete categories. You can create one using `plt.bar()`.

```python
labels = ['A', 'B', 'C']
values = [1, 4, 2]

bars = plt.bar(labels, values)

# Add patterns to the bars
bars[0].set_hatch('/')
bars[1].set_hatch('o')
bars[2].set_hatch('*')

plt.show()
```

## 6. Saving Your Plot

It's great to see your plot in the notebook, but you'll often need to save it as a file. You can do this with `plt.savefig()`.

It's recommended to save the figure *before* you show it.

```python
plt.plot(x, y)
# ... (all your other plot customizations)

# Save the figure. dpi (dots per inch) controls the resolution.
plt.savefig('mygraph.png', dpi=300)

plt.show()
```
This concludes our introduction to Matplotlib. You now have the foundational skills to create, customize, and save basic plots in Python.
