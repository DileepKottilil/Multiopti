import matplotlib.pyplot as plt
import numpy as np

y1 = 0
lyr_1 = 10
lyr_2 = 5
plt.axes()

for i in np.arange(2):
    
    rectangle1 = plt.Rectangle((0,-y1), 50, -lyr_1, fc='blue',ec="red")
    rectangle2 = plt.Rectangle((0,-y1-lyr_1), 50, -lyr_2, fc='white',ec="black")

    plt.gca().add_patch(rectangle1)
    plt.gca().add_patch(rectangle2)
    y1 = lyr_1+lyr_2

    plt.axis('scaled')
    
plt.tight_layout
plt.show()