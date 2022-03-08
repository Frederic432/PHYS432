# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
import matplotlib.pyplot as pl

# Set up the grid and advection parameters
Ngrid = 50
Nsteps = 100
dt = 1
dx = 1

v = -0.1
alpha = v * dt / 2 / dx

x = np.arange(Ngrid)

f1 = np.copy(x) * 1. / Ngrid
f2 = np.copy(x) * 1. / Ngrid

# Set up plot
pl.ion()
fig, axes = pl.subplots(1, 2)
axes[0].set_title('FTC')
axes[0].set_title('lax')

# background reference
axes[0].plot(x, f1, 'k-')
axes[1].plot(x, f2, 'k-')

# We will be updating this plotting object
plt1, = axes[0].plot(x, f1, 'ro')
plt2, = axes[1].plot(x, f1, 'ro')

# Setting the axis limits for visualization
for ax in axes:
    ax.set_xlim([0, Ngrid])
    ax.set_ylim([0, 2])

# this draws the objects on the plot
fig.canvas.draw()

count = 0

# Evolution
while count < Nsteps:
    # FTC
    f1[1:Ngrid - 1] = f1[1:Ngrid - 1] - alpha * (f1[2:] - f1[:Ngrid - 2])
    # LAX
    # f2[1:Ngrid-1] = 0.5*(f2[2:] + f2[:Ngrid-2]) - alpha*(f2[2:] - f2[:Ngrid-2])
    # Godunov
    upwind = f2[1:Ngrid - 1] - 2 * alpha * (f2[1:Ngrid - 1] - f2[:Ngrid - 2])
    downwind = f2[1:Ngrid - 1] - 2 * alpha * (f2[2:] - f2[1:Ngrid - 1])
    f2[1:Ngrid - 1] = np.where(v > 0, upwind, downwind)

    plt1.set_ydata(f1)
    plt2.set_ydata(f2)

    fig.canvas.draw()
    pl.pause(0.1)
    count += 1
