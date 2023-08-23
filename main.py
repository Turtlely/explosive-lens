# Simulate the propogation of an explosive pressure wave within a heterogenous medium

# Shock waves obey snell's law

# Simple case of a shock wave in one medium meeting another medium with a simple straight line boundary

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 
import matplotlib.animation as animation
import matplotlib.path as mpltPath
import itertools

class particle:
    def __init__(self,x,y,v,theta):
        self.v = v
        self.theta=theta
        self.theta0 = theta
        self.x = x
        self.y = y
        self.flag = True
        self.color = 'blue'
    
    def update_position(self,dt):
        self.x += self.v*np.cos(self.theta)*dt
        self.y += self.v*np.sin(self.theta)*dt

# Numerical sim constants
dt = 0.01
num_particles = 100

# Simulation constants
v1 = 2
v2 = 1
t_d = 3.5
d = 4
r = 2.4

# Set up figure
fig = plt.figure()

# Plot explosive lens geometry
delta = 0.0025
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

path = mpltPath.Path([(a,b) for (a,b) in zip(cont_x_points,cont_y_points)])

dydx = np.nan_to_num(np.gradient(cont_y_points,cont_x_points),True,np.inf)

# Make list of particles for the simulation
plist = [particle(0,0,v1, theta) for theta in np.linspace(-0.5415,0.5415,num_particles+1)]
#plist = [particle(0,0,v1, theta) for theta in np.linspace(-np.pi/2,np.pi/2,num_particles+1)]

#plist = [particle(0,0,v1,0)]

# Update the positions of the particles

circle = plt.Circle((d, 0), r, color='b', fill=False)

def animate(i):
    print(i)
    plt.cla()
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.xlim([0, 5])
    plt.ylim([-2, 2])

    plt.gca().add_patch(circle)

    x = []
    y = []
    c = []

    # For every point    
    for p in plist:
        c.append(p.color)
        # Is the point inside the explosive lens?

        if p.flag:
            index = np.argmin(np.abs(np.array(cont_y_points)-p.y))
            
            if p.x > cont_x_points[index]:
            #if path.contains_point((p.x,p.y)):
                p.color = 'hotpink'
                p.v = v2

                # Find new angle of the particle

                # Slope of lens at intercept
                m = dydx[index]

                angle_incidence = p.theta0-np.arctan(-1/m)

                angle_output = np.arcsin(v2*np.sin(angle_incidence)/v1)

                final_angle = np.arctan(-1/m)+angle_output

                p.theta = final_angle
                p.flag = False

        p.update_position(dt)
        x.append(p.x)
        y.append(p.y)

    plt.plot(x,y)
    plt.plot(cont_x_points,cont_y_points)
    plt.scatter(d,0)


anim = FuncAnimation(fig,animate,interval = 1,frames=105)
fig.suptitle('Detonation wave in an explosive lens', fontsize=14)
writervideo = animation.FFMpegWriter(fps=15)
anim.save('detonation.gif', writer=writervideo)
plt.show()