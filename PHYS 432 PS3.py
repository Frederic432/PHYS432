"""
PHYS432 W2022 Problem Set 3
Advection-Diffusion of Lava Flowing Down an Inclined Plane

@author: Frédéric Duong
Mar. 10th 2022
"""

import numpy as np
import matplotlib.pyplot as plt

# all units are in SI units
# grid and advection and diffusion parameters
Ngrid = 50
Nsteps = 10000
dt1 = 2.1
dt2 = 0.001
dx = 1

v = -0.01 # advection velocity
alpha = v*dt2/dx
nu = 20 # the diffusion coefficient is the kinematic viscosity
beta = nu*dt1/dx**2

# taking gravity into account for the advection
g = 9.8
theta = (20*np.pi)/180 # choosing an angle of 20 degrees for the inclined plane
gamma = g*np.sin(theta)*dt2

x = np.arange(0, Ngrid * 1., dx) / Ngrid  # multiplying by 1. to make sure this is an array of floats not integers
                                          # the grid runs from 0 to 1
                                          # this also sets the height of the lava to 1

# setting up the initial shape of the lava layers
f = np.copy(x)

# set up the plot
plt.ion()
fig, axes = plt.subplots(1,1)
axes.set_title('Velocity of lava flow as a function of its height')

# plotting initial velocity's magnitude in the background for reference
axes.plot(x, f, 'k-', label = 'Initial velocity')

# plotting the final velocity's magnitude in the background for reference
axes.plot(x, (g*np.sin(theta)/(nu))*(-(1/2)*x**2 + x), 'b', label = 'Analytical solution')
plt.legend()

# this plot will be updated
plt1, = axes.plot(x, f, 'ro')

# x and y axes limits of the plot
axes.set_xlim([0,1])
axes.set_ylim([0,1])

plt.xlabel('Height of the lava (m)')
plt.ylabel('Velocity (m/s)')

# drawing the velocity's magnitude on the plot
fig.canvas.draw()

for ct in range(Nsteps):

    ## calculate diffusion first
    # setting up matrices for diffusion operator
    A = np.eye(Ngrid) * (1.0 + 2.0 * beta) + np.eye(Ngrid, k=1) * -beta + np.eye(Ngrid, k=-1) * -beta

    ## no-slip boundary condition to keep the velocity of the lowest layer fixed
    A[0][0] = 1.0
    A[0][1] = 0

    ## no-stress boundary condition at the air-lava interface
    A[Ngrid-1][Ngrid-1] = 1.0 + beta

    # solving for the next time step
    f = np.linalg.solve(A, f)

    ## Calculate advection (Godunov method)
    f[1:Ngrid - 1] = np.where(v > 0, f[1:Ngrid-1] - alpha*(f[1:Ngrid-1] - f[:Ngrid-2]) + gamma, f[1:Ngrid-1] - alpha*(f[2:] - f[1:Ngrid-1]) + gamma)

    # updating the plot
    plt1.set_ydata(f)

    fig.canvas.draw()
    plt.pause(0.001)
