"""
Exercise 2: Solutions - Data Visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

# Load sample data
tips = sns.load_dataset('tips')

# Solution 1: Customized Matplotlib plot
plt.figure(figsize=(10, 6))
for day in tips['day'].unique():
    day_data = tips[tips['day'] == day]
    plt.scatter(day_data['total_bill'], 
                day_data['tip'], 
                label=day,
                alpha=0.7)
plt.title('Tips vs Total Bill by Day')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.legend(title='Day')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.close()

# Solution 2: Seaborn box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', hue='time', data=tips)
plt.title('Distribution of Total Bill by Day and Time')
plt.xlabel('Day of Week')
plt.ylabel('Total Bill ($)')
plt.tight_layout()
plt.savefig('box_plot.png')
plt.close()

# Solution 3: Interactive Plotly visualization
fig = px.scatter(
    tips,
    x='total_bill',
    y='tip',
    color='smoker',
    size='size',
    hover_data=['sex', 'day', 'time'],
    title='Interactive Tips Visualization',
    labels={'total_bill': 'Total Bill ($)', 'tip': 'Tip ($)'}
)
fig.update_layout(showlegend=True)
fig.write_html('interactive_plot.html')
