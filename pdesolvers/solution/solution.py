import pdesolvers.pdes.black_scholes as bse
import numpy as np
from matplotlib import pyplot as plt

class Solution1D:

    def __init__(self, result, x_grid, t_grid):
        self.result = result
        self.x_grid = x_grid
        self.t_grid = t_grid

    def plot(self):

        if self.result is None:
            raise RuntimeError("Solution has not been computed - please run the solver.")

        x_plot, t_plot = np.meshgrid(self.x_grid,self.t_grid)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_plot, t_plot, self.result, cmap='viridis')

        # set labels and title
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        ax.set_title('3D Surface Plot of 1D Heat Equation')

        plt.show()

    def get_result(self):
        return self.result

class SolutionBlackScholes:
    def __init__(self, result, s_grid, t_grid):
        self.result = result
        self.s_grid = s_grid
        self.t_grid = t_grid

    def plot(self):

        X, Y = np.meshgrid(self.t_grid, self.s_grid)

        # plotting the 3d surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.result, cmap='viridis')

        ax.set_xlabel('Time')
        ax.set_ylabel('Asset Price')
        ax.set_zlabel('Option Value')
        ax.set_title('Option Value Surface Plot')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def get_result(self):
        return self.result