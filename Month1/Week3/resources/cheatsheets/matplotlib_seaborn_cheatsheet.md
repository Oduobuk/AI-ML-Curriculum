# Matplotlib & Seaborn Cheat Sheet

## Matplotlib Basics

### Import and Setup
```python
import matplotlib.pyplot as plt
import numpy as np

# Enable inline plotting in Jupyter
%matplotlib inline

# Set style
plt.style.use('seaborn')

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))
```

### Basic Plots
```python
# Line plot
plt.plot(x, y, label='line', color='blue', linestyle='-', linewidth=2, marker='o')

# Scatter plot
plt.scatter(x, y, c='red', s=50, alpha=0.5, label='points')

# Bar plot
plt.bar(x, height, width=0.8, align='center', color='green')

# Histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Box plot
plt.boxplot(data, vert=True, patch_artist=True)
```

### Customization
```python
# Titles and labels
plt.title('Title', fontsize=14)
plt.xlabel('X Label', fontsize=12)
plt.ylabel('Y Label', fontsize=12)
plt.legend(loc='best')

# Axis limits and ticks
plt.xlim(0, 10)
plt.ylim(0, 100)
plt.xticks([0, 5, 10], ['zero', 'five', 'ten'])

# Grid and layout
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

## Seaborn Basics

### Import and Setup
```python
import seaborn as sns

# Set style and context
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.2)
```

### Distribution Plots
```python
# Distribution plot
sns.distplot(data, kde=True, bins=30)

# Kernel Density Estimate
sns.kdeplot(data, shade=True)

# Joint plot (scatter + histograms)
sns.jointplot(x='x', y='y', data=df, kind='scatter')

# Pair plot
sns.pairplot(df, hue='category')
```

### Categorical Plots
```python
# Bar plot
sns.barplot(x='x', y='y', data=df, hue='category')

# Count plot
sns.countplot(x='category', data=df)

# Box plot
sns.boxplot(x='category', y='value', data=df)

# Violin plot
sns.violinplot(x='category', y='value', data=df, inner='quartile')

# Swarm plot
sns.swarmplot(x='category', y='value', data=df, color='black', alpha=0.5)
```

### Matrix Plots
```python
# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Clustermap
sns.clustermap(corr_matrix, cmap='coolwarm')
```

### Regression Plots
```python
# Linear regression
sns.lmplot(x='x', y='y', data=df, hue='category')

# Regression with polynomial fits
sns.regplot(x='x', y='y', data=df, order=2)
```

### Facet Grids
```python
# Create a grid of plots
g = sns.FacetGrid(df, col='category', col_wrap=3)
g.map(plt.hist, 'value')

# With custom function
g = sns.FacetGrid(df, row='category', col='time')
g.map(sns.regplot, 'x', 'y')
```

### Styling
```python
# Color palettes
sns.color_palette('husl', 8)
sns.palplot(sns.color_palette('husl'))

# Set context (paper, notebook, talk, poster)
sns.set_context('talk')

# Set style (darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')
```

## Advanced Customization

### Multiple Plots
```python
# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot on first axis
ax1.plot(x, y1, 'r-')
ax1.set_title('First Plot')

# Plot on second axis
ax2.scatter(x, y2, c='blue', alpha=0.5)
ax2.set_title('Second Plot')

# Adjust layout
plt.tight_layout()
```

### Annotations
```python
# Add text
plt.text(x, y, 'Important Point', fontsize=12)

# Add arrow
plt.annotate('Peak', xy=(x_peak, y_peak), xytext=(x_text, y_text),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Add vertical/horizontal lines
plt.axvline(x=0, color='k', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
```

### Saving Figures
```python
# Save as PNG with high DPI
plt.savefig('figure.png', dpi=300, bbox_inches='tight')

# Save as PDF (vector format)
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')

# Save transparent background
plt.savefig('figure.png', transparent=True, dpi=300)
```

## Common Plot Types

### Time Series
```python
# Basic time series
plt.plot_date(dates, values, '-')

# With error bands
plt.fill_between(dates, values - std, values + std, alpha=0.2)
```

### Pie Charts
```python
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures circular pie
```

### Error Bars
```python
plt.errorbar(x, y, yerr=error, fmt='o', markersize=5, capsize=5)
```

### 3D Plots
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
```

## Tips and Tricks

### Plotting from Pandas
```python
# Direct plotting from DataFrame
df.plot(kind='line', x='date', y='value')
df['column'].plot(kind='hist', bins=30)

# Multiple columns
df[['A', 'B', 'C']].plot(subplots=True, layout=(3, 1))
```

### Interactive Plots
```python
# Enable interactive mode
%matplotlib notebook

# Create interactive plot
plt.ion()
plt.plot(x, y)
plt.ioff()
```

### LaTeX in Plots
```python
plt.title(r'$\alpha > \beta$')
plt.xlabel('$\mu=100,\ \sigma=15$')
```

### Tight Layout
```python
# Automatically adjust subplot params
plt.tight_layout()

# Or with custom padding
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
```
