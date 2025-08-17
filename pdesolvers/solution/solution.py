from matplotlib.animation import FuncAnimation
import numpy as np
import pdesolvers.utils.utility as utility
import pdesolvers.enums.enums as enum

from matplotlib import pyplot as plt

import logging

logging.basicConfig(
    level = logging.INFO,
    format = "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

class Solution:

    def __init__(self, result, x_grid, y_grid, dx, dy, duration):
        self.result = result
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.dx = dx
        self.dy = dy
        self.duration = duration

    def plot(self, export=False):
        """
        Generates a 3D surface plot of the option values across a grid of asset prices and time

        :return: 3D surface plot
        """

        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.size'] = 10

        X, Y = np.meshgrid(self.x_grid, self.y_grid)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.result, cmap='plasma', alpha=0.8)

        self._set_plot_labels(ax)

        if export:
            plt.savefig("solution_results.pdf", format="pdf", bbox_inches="tight", pad_inches=0.5)

        plt.show()

    def get_result(self):
        """
        Gets the grid of computed temperature values

        :return: grid result
        """
        return self.result

    def _set_plot_labels(self, ax):
        """
        Sets labels for the plot
        :param ax: axes
        """
        # font = {'family': 'serif'}
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Value')
        ax.set_title('3D Surface Plot')

    def get_execution_time(self):
        """
        Gets the time taken for the solver to solve the equation
        :return: duration
        """
        return self.duration

    def __sub__(self, other):
        """
        Compares two solutions by interpolating the sparse grid to the dense grid and computing the difference
        :param other: the grid to be compared against
        :return: the error (difference) between the two solutions
        """

        logging.info("Interpolating solutions with RBFInterpolator")

        sparser_grid = other
        denser_grid = self

        if self.x_grid.shape[0] < other.x_grid.shape[0] and self.y_grid.shape[0] < other.y_grid.shape[0]:
            sparser_grid = self
            denser_grid = other

        interpolator_sparse = utility.RBFInterpolator(sparser_grid.result.T, sparser_grid.dx, sparser_grid.dy)

        diff = 0

        for idx_x in range(denser_grid.x_grid.shape[0]):
            for idx_y in range(denser_grid.y_grid.shape[0]):

                # Points (x, y) of the dense grid
                x = denser_grid.x_grid[idx_x]
                y = denser_grid.y_grid[idx_y]

                # Value at (x, y) for the dense grid
                val_dense_x_y = denser_grid.result[idx_y, idx_x]

                # Interpolate the sparse grid at (x, y)
                val_sparse_x_y = interpolator_sparse.interpolate(x, y)

                diff = np.max([diff, np.abs(val_dense_x_y - val_sparse_x_y)])

        logging.info("Interpolation completed.")

        return diff

class Solution1D(Solution):

    def __init__(self, result, x_grid, y_grid, dx, dy, duration):
        super().__init__(result, x_grid, y_grid, dx, dy, duration)

    def _set_plot_labels(self, ax):
        super()._set_plot_labels(ax)
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        ax.set_title('3D Surface Plot of 1D Heat Equation')


class SolutionBlackScholes(Solution):
    def __init__(self, result, x_grid, y_grid, dx, dy, duration, delta, gamma, theta, option_type):
        super().__init__(result, x_grid, y_grid, dx, dy, duration)
        self.option_type = option_type
        self.delta = delta
        self.gamma = gamma
        self.theta = theta

    def plot_greek(self, greek_type=enum.Greeks.DELTA, time_step=0, export=False):

        greek_types = {
            enum.Greeks.DELTA : {'data': self.delta, 'title': enum.Greeks.DELTA.value},
            enum.Greeks.GAMMA : {'data': self.gamma, 'title': enum.Greeks.GAMMA.value},
            enum.Greeks.THETA : {'data': self.theta, 'title': enum.Greeks.THETA.value}
        }

        # if greek_type.lower() not in greek_types:
        #     raise ValueError("Invalid greek type - please choose between delta/gamma/theta.")

        chosen_greek = greek_types[greek_type]
        greek_data = chosen_greek['data'][:, time_step]
        plt.figure(figsize=(8, 6))
        plt.plot(self.y_grid, greek_data, label=f"{chosen_greek['title']} at t={self.x_grid[time_step]:.4f}", color="grey", alpha=0.8)

        plt.title(f"{chosen_greek['title']} vs. Stock Price at t={self.x_grid[time_step]:.4f}")
        plt.xlabel("Stock Price (S)")
        plt.ylabel(chosen_greek['title'])
        plt.grid()
        plt.legend()

        if export:
            np.savetxt(f"option_{chosen_greek['title']}.csv", np.column_stack((self.y_grid, greek_data)), delimiter=',', header=f"StockPrice,{chosen_greek['title']}", comments='')

        plt.show()

    def _set_plot_labels(self, ax):
        super()._set_plot_labels(ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Asset Price')
        ax.set_zlabel(f'{self.option_type.value} Option Value')
        ax.set_title(f'{self.option_type.value} Option Value Surface Plot')

class Heat2DSolution:
    def __init__(self, result, x_grid, y_grid, t_grid, dx, dy, dt, duration):
        self.result = result
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.t_grid = t_grid
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.duration = duration

    @staticmethod
    def __plot_surface(u_k, k, ax, xDomain, yDomain, dt):
        ax.clear()
        X, Y = np.meshgrid(xDomain, yDomain)
        # Transpose u_k to match meshgrid orientation
        surf = ax.plot_surface(X, Y, u_k.T, 
                            cmap='hot',
                            alpha=0.9)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position') 
        ax.set_zlabel('Temperature')
        ax.set_title(f'2D Heat Equation: t = {k*dt:.4f}')
        ax.set_zlim(0, 50)

        return surf
    
    def animate(self, export=False, filename="heat_equation_2d_plot.gif"):
        print("Creating animation...")
        self
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animateFrame(k):
            return Heat2DSolution.__plot_surface(self.result[k], k, ax, self.x_grid, self.y_grid, self.dt)
        
        anim = FuncAnimation(fig, animateFrame, interval=100, frames=len(self.t_grid), repeat=True)
        
        if export:
            anim.save(filename+'.gif', writer='pillow', fps=5)

        plt.show()
    
    def get_result(self):
        """
        Gets the grid of computed temperature values

        :return: grid result
        """
        return self.result
        
    def get_execution_time(self):
        """
        Gets the time taken for the solver to solve the equation
        :return: duration
        """
        return self.duration
    