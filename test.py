

from matplotlib import pyplot as plt
import numpy as np
delta = 0.025
xrange = np.arange(0.0, 6.0, delta)
yrange = np.arange(-5.0, 5.0, delta)
x, y = np.meshgrid(xrange, yrange)

v1 = 2
v2 = 1
t_d = 3.5
d = 4

equation = np.sqrt(x**2+y**2)/v1 + np.sqrt((d-x)**2+y**2)/v2 - t_d
plt.contour(x, y, equation, [0])
plt.gca().set_aspect('equal')
plt.grid()
plt.show()