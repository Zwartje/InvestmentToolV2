import matplotlib.pyplot as plt
import numpy as np

def generate_figure(index):
    x = np.random.rand(50)
    y = np.random.rand(50)
    plt.scatter(x, y)
    plt.title(f'Scatter Plot {index}')

# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2)

# Generate scatter plots and plot them in the subplots
for i, ax in enumerate(axs.flatten(), 1):
    generate_figure(i)
    plt.sca(ax)

# Adjust the layout to prevent overlapping titles
plt.tight_layout()

# Show the subplots
plt.show()
