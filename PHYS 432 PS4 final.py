"""
Evolving a strong adiabatic shock

@author: Frédéric Duong
March 24th 2022
"""
import numpy as np
import matplotlib.pyplot as plt

# Set up the grid, time and grid spacing, and the sound speed squared
Ngrid = 100
Nsteps = 5000
dt = 0.1
dx = 2
gamma = 5 / 3  # adiabatic index

x = np.arange(Ngrid) * dx  # grid
f1 = np.ones(Ngrid)  # rho, initial density = 1
f2 = np.zeros(Ngrid)  # rho x u, initial momentum = 0 since initial velocity = 0
f3 = np.ones(Ngrid)  # rho x energy, initial energy density = 1
u = np.zeros(Ngrid+1)  # advective velocity (keep the 1st and last element zero)


def advection(f, u, dt, dx):
    # calculating flux terms
    J = np.zeros(len(f) + 1)  # keeping the first and the last term zero
    J[1:-1] = np.where(u[1:-1] > 0, f[:-1] * u[1:-1], f[1:] * u[1:-1])
    f = f - (dt / dx) * (J[1:] - J[:-1])  # update

    return f

# Apply initial Gaussian perturbation
Amp, sigma = 4, Ngrid / 10
f1 = f1 + Amp * np.exp(-(x - x.max() / 2) ** 2 / sigma ** 2)
f2 = f2 + Amp * np.exp(-(x - x.max() / 2) ** 2 / sigma ** 2)
f3 = f3 + Amp * np.exp(-(x - x.max() / 2) ** 2 / sigma ** 2)

# plotting
plt.ion()
fig, axes = plt.subplots(1,2)
axes[0].set_title('Density plot')
axes[1].set_title('Mach number')

# We will be updating these plotting objects
plt1, = axes[0].plot(x, f1, 'ro', markersize='2')
plt2, = axes[1].plot(x, u[:-1]**2/(gamma*f1**(gamma-1)), 'ro', markersize='2')

# Setting the axis limits for visualization
axes[0].set_xlim([0, dx * Ngrid + 1])
axes[1].set_xlim([0, dx * Ngrid + 1])
axes[0].set_ylim([-1, 5])
axes[1].set_ylim([-1, 5])

axes[0].set_xlabel('x')
axes[1].set_xlabel('x')
axes[0].set_ylabel('Density (kg/m^3)')
axes[1].set_ylabel('Mach number')

fig.canvas.draw()

for ct in range(Nsteps):
    # advection velocity at the cell interface
    u[1:-1] = 0.5 * ((f2[:-1] / f1[:-1]) + (f2[1:] / f1[1:]))

    # advect density and momentum
    f1 = advection(f1, u, dt, dx)
    f2 = advection(f2, u, dt, dx)

    # pressure = rho**gamma for adiabatic process
    P = f1 ** gamma

    # add the source term to momentum
    f2[1:-1] = f2[1:-1] - 0.5 * (dt / dx) * (P[2:] - P[:-2])

    # correct for source term at the boundary (reflective)
    f2[0] = f2[0] - 0.5 * (dt / dx) * (P[1] - P[0])
    f2[-1] = f2[-1] - 0.5 * (dt / dx) * (P[-1] - P[-2])

    # re-advection of velocity at the cell interface
    u[1:-1] = 0.5 * ((f2[:-1] / f1[:-1]) + (f2[1:] / f1[1:]))

    # advect energy
    f3 = advection(f3, u, dt, dx)

    # pressure
    P = f1 ** gamma

    # add the source term to energy density
    # the source is the negative divergence of pressure times momentum
    f3[1:-1] = f3[1:-1] - 0.5 * (dt / dx) * (P[2:] * f2[2:] - P[:-2] * f2[:-2])

    # correct for source term at the boundary (reflective)
    f3[0] = f3[0] - 0.5 * (dt / dx) * (P[1] * f2[1] - P[0] * f2[0])
    f3[-1] = f3[-1] - 0.5 * (dt / dx) * (P[-1] * f2[-1] - P[-2] * f2[-2])

    # pressure
    P = f1 ** gamma

    # sound speed squared = gamma*pressure/density
    cs2 = gamma * f1 ** (gamma - 1)

    # Mach number
    M = u[:-1]**2/cs2

    # update the plot
    plt1.set_ydata(f1)
    plt2.set_ydata(M)
    fig.canvas.draw()
    plt.pause(0.001)
