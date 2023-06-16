import matplotlib.pyplot as plt

import numpy as np

# Load the data
data = np.loadtxt('ref_indx.txt', skiprows = 1)

# Assign columns to variables
x = data[:, 0]
y1 = data[:, 1]
y2 = data[:, 2]

# Now you can use x, y1, and y2

# Create the plot
plt.figure(figsize=(10,6))

# Plot y1 vs x
plt.plot(x, y1, label='Column 2')

# Plot y2 vs x
plt.plot(x, y2, label='Column 3')

# Add a legend
plt.legend()

# Show the plot
plt.show()
