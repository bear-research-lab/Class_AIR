import numpy as np
import matplotlib.pyplot as plt

# Data from the slide
colors = ['Red', 'Yellow', 'Green']
sweetness = np.array([10, 8, 6])
probabilities = np.array([0.6, 0.1, 0.3])

# Calculate Expectation: E = Sum of (P(x) * f(x))
expected_sweetness = np.sum(probabilities * sweetness)

# Visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the individual sweetness values
bars = ax.bar(colors, sweetness, color=[
              'red', 'gold', 'green'], alpha=0.6, label='Sweetness f(x)')

# Add probability labels on top of bars
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'P={probabilities[i]*100}%', ha='center', fontweight='bold')

# Draw the Expectation line
ax.axhline(expected_sweetness, color='blue', linestyle='--', linewidth=2,
           label=f'Expected Sweetness (Avg) = {expected_sweetness:.2f}')

ax.set_ylim(0, 12)
ax.set_ylabel('Sweetness Level')
ax.set_title('Expectation: Weighted Average of Apple Sweetness')
ax.legend()

plt.show()
