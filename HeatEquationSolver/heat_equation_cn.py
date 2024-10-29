import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# setting up parameters
length = 1
nodes = 100
time = 30
k = 0.01
initial_temp = 25


def initialise_grid(length, nodes, time, k):
    """
    Initialises the grid with the desired parameters

    :param length: length of the rod
    :param nodes: number of spatial nodes across the rod
    :param time:  the total time
    :param k: heat diffusivity constant
    :return: parameters to set up the grid (dx, dt, time_step, x, t)
    """
    dx = length / (nodes-1)
    dt_max = 0.5 * (dx**2) / k  # calculating dt using a formula (to ensure stability)
    dt = 0.8 * dt_max
    time_step = int(time/dt)
    x = np.linspace(0, length, nodes)
    t = np.linspace(0, time, time_step)

    return dx, dt, time_step, x, t


def left_temp_func(temp, t):
    """
    Generates the temperature at the left boundary of the rod

    :param temp: initial temperature of the rod at t=0
    :param t: array consisting of time steps from t=0 to t= total time
    :return: the temperature at the left boundary
    """
    return 20 * np.sin(np.pi * t) + temp


def right_temp_func(temp, t):
    """
    Generates the temperature at the right boundary of the rod

    :param temp: initial temperature of the rod at t=0
    :param t: array consisting of time steps from t=0 to t= total time
    :return: the temperature at the right boundary
    """
    return t + temp


def setting_conditions(t, time_step, nodes, temp):
    """
    Initialises the temperature distribution and boundary conditions for the heat equation

    :param t: array consisting of time steps from t=0 to t=total time
    :param time_step: no of time steps for the simulation
    :param nodes: no of spatial nodes
    :param temp: initial temperature of the rod at t=0
    :return:    - u: 2d-matrix initialised with the initial temperature of the rod
                - u_left: array denoting the temperature at the left boundary over time
                - u_right: array denoting the temperature at the right boundary over time
    """
    u_initial = temp
    u_left = left_temp_func(temp, t)
    u_right = right_temp_func(temp, t)
    u = np.zeros((time_step, nodes))
    u[0, :] = u_initial

    return u, u_left, u_right


def solve_heat_equation(u, u_left, u_right, time_step, nodes, dt, k, dx):
    """
    Solves the heat equation using the explicit scheme

    :param u: 2d-matrix initialised with the initial temperature of the rod
    :param u_left: array denoting the temperature at the left boundary over time
    :param u_right: array denoting the temperature at the right boundary over time
    :param time_step: no of time steps for the simulation
    :param nodes: no of spatial nodes
    :param dt: time step size
    :param k: heat diffusivity constant
    :param dx: spatial step size
    :return:    - u: the solved 2d matrix consisting the temperature distribution over time
    """
    for tau in range(0, time_step-1):
        u[tau+1, 0] = u_left[tau+1]
        u[tau+1, -1] = u_right[tau+1]
        for i in range(1, nodes - 1):
            u[tau+1,i] = u[tau, i] + (dt * k * (u[tau, i-1] - 2 * u[tau, i] + u[tau, i+1]) / dx**2)

    return u


def plot(x, u, dt, time_step,):
    """
    Plots the temperature distribution of the rod over time

    :param x: array consisting of spatial steps from t=0 to t=length
    :param u: the solved 2d matrix consisting the temperature distribution over time
    :param dt: time step size
    :param time_step: no of time steps for the simulation
    :return:
    """
    # setting up plots
    fig, ax = plt.subplots()
    ax.set_title('Heat Distribution over time')
    ax.set_xlabel('Length across the rod')
    ax.set_ylabel('Temperature')
    # plotting initial temperature distribution at t = 0
    line, = ax.plot(x, u[0])
    # setting y-axis limits
    ax.set_ylim(bottom=0, top=60)

    def animate(frame):
        line.set_ydata(u[frame])
        ax.set_title(f'Heat Distribution at time = {frame * dt:.2f}s')

        return line,

    ani = FuncAnimation(fig, animate, frames=range(0, time_step, time_step // 100), blit=False)
    plt.show()


def main():
    dx, dt, time_step, x, t = initialise_grid(length, nodes, time, k)
    u, u_left, u_right = setting_conditions(t, time_step, nodes, initial_temp)
    u = solve_heat_equation(u, u_left, u_right, time_step, nodes, dt, k, dx)
    print(u)
    plot(x, u, dt, time_step)


if __name__ == "__main__":
    main()

