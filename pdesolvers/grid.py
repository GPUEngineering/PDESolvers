import inspect
import numpy as np


class FDMGrid:
    """
    A grid for solving PDEs with finite differences methods
    """

    def __init__(self):
        self.__grid = None
        self.__values = None
        self.__mask = None
        self.__geometry = None

        # Make this constructor private
        caller_frame = inspect.stack()[1]
        _caller_module = inspect.getmodule(caller_frame[0])
        caller_name = caller_frame.function
        assert caller_name.startswith('create'), 'use factory methods instead'

    @classmethod
    def create_uniform_grid(cls, x_from, x_to, n_points, coordinate_labels=None):
        """
        Create a uniform grid on a rectangular geometry. The boundaries of the geometry are specified by the arrays
        x_from and x_to. The number of points along each direction are given in n_points.

        Parameters
        ----------
        :param x_from: starting positions   (array)
        :param x_to: end positions  (array)
        :param n_points: number of points (array)
        :param coordinate_labels: coordinate labels, e.g., ["x", "y", "t"]
        :return: instance of FDMGrid

        Notes
        ----------
        The values on the grid are initialized with zeros

        Example
        ----------
        For example, to make a 3D grid over the rectangle [0, 1] x [0, 2] x [-3, 3] with 10, 20, and 30 points in each
        direction (coordinate), do

        >>> x_start = np.array([0, 0, 0])
        >>> x_end = np.array([1, 2, 3])
        >>> num_points = np.array([10, 20, 31])
        >>> unif_grid = FDMGrid.create_uniform_grid(x_start, x_end, num_points)
        """
        gr = FDMGrid()
        n = x_from.size
        gr.__grid = np.meshgrid(*[np.linspace(x_from[i], x_to[i], n_points[i]) for i in range(n)])
        gr.__values = np.zeros(n_points)
        gr.__geometry = {"geometry": "rectangular",
                         "grid_type": "uniform",
                         "from": x_from,
                         "to": x_to}
        if coordinate_labels is not None:
            gr.__geometry["coordinate_labels"] = coordinate_labels
        return gr

    @classmethod
    def create_dense_at_boundary_grid(cls, x_from, x_to, n_points, density_boosting=2.0, coordinate_labels=None):
        pass

    @classmethod
    def create_grid_nonsquare_geometry(cls):
        pass

    def values(self):
        return self.__values

    def mesh(self):
        return self.__grid

    def mask(self):
        return self.__mask

    def initialize(self):
        """
        Set initial and boundary conditions
        """
        pass

    def plot(self):
        # Plot the values on your grid
        # You'll need a different implementation for different dimensions
        # The user could be given the option to see a plot or an animation
        # The plot should be configurable by the user
        pass


