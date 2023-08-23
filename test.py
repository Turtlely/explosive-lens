import numpy as np
import matplotlib.pyplot as plt

# Simulation constants
v1 = 2
v2 = 1
t_d = 3.5
d = 4
r = 2.35

delta = 0.025

xrange = np.arange(0.0, 6.0, delta)
yrange = np.arange(-5.0, 5.0, delta)
xe, ye = np.meshgrid(xrange, yrange)

equation = np.sqrt(xe**2+ye**2)/v1 + np.sqrt((d-xe)**2+ye**2)/v2 - t_d


contour = plt.contour(xe, ye, equation, [0])
cont_x_points = []
cont_y_points = []
for item in contour.collections:
        for i in item.get_paths():
            v = i.vertices
            cont_x_points = v[:, 0]
            cont_y_points = v[:, 1]

dydx = np.gradient(cont_y_points,cont_x_points)

index = np.argmin(np.abs(np.array(cont_y_points)-(-1)))

print(dydx[index])


plt.plot(cont_x_points,cont_y_points)
plt.gca().set_aspect('equal')
plt.grid()
plt.show()

