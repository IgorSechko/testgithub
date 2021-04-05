import numpy as np
import matplotlib.pyplot as plt
import torch

cutoffs, cutoff_layer = torch.load("tensors/adaptive_layers.pt")
# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(111, projection='3d')

# fake data
_x = np.linspace(0, 5, 5)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 0.5

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')

plt.show()
