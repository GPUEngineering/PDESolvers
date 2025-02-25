import numpy as np


class RBFInterpolator:

    def __init__(self, z, x, y, hx, hy):
        """
        Initializes the RBF Interpolator.

        :param z: 2D array of values at the grid points.
        :param x: x-coordinate of the point to interpolate.
        :param y: y-coordinate of the point to interpolate.
        :param hx: Grid spacing in the x-direction.
        :param hy: Grid spacing in the y-direction.
        """

        self._z = z
        self._x = x
        self._y = y
        self._hx = hx
        self._hy = hy
        self._nx, self._ny = z.shape

    def _get_coordinates(self):
        """
        Determines the x and y coordinates of the bottom-left corner of the grid cell

        :return: A tuple containing the coordinates and its corresponding indices
        """

        # gets the grid steps to x
        i_minus_star = int(np.floor(self._x / self._hx))
        if i_minus_star > self._nx - 1:
            raise Exception("x is out of bounds")

        # final i index for interpolation
        i_minus = i_minus_star if i_minus_star < self._nx - 1 else self._nx - 1

        # gets the grid steps to y
        j_minus_star = int(np.floor(self._y / self._hy))
        if j_minus_star > self._ny - 1:
            raise Exception("y is out of bounds")

        # final j index for interpolation
        j_minus = j_minus_star if j_minus_star < self._ny - 1 else self._ny - 1

        # computes the coordinates at the computed indices
        x_minus = i_minus * self._hx
        y_minus = j_minus * self._hy

        return x_minus, y_minus, i_minus, j_minus

    def _euclidean_distances(self, x_minus, y_minus):
        """
        Calculates Euclidean distances between (x,y) and the surrounding grid points in the unit cell

        :param x_minus: x-coordinate of the bottom-left corner of the grid
        :param y_minus: y-coordinate of the bottom-left corner of the grid
        :return: returns tuple with the Euclidean distances to the surrounding grid points:
                [bottom left, top left, bottom right, top right]
        """

        bottom_left = np.sqrt((x_minus - self._x) ** 2 + (y_minus - self._y) ** 2)
        top_left = np.sqrt((x_minus - self._x) ** 2 + (y_minus + self._hy - self._y) ** 2)
        bottom_right = np.sqrt((x_minus + self._hx - self._x) ** 2 + (y_minus - self._y) ** 2)
        top_right = np.sqrt((x_minus + self._hx - self._x) ** 2 + (y_minus + self._hy - self._y) ** 2)

        return bottom_left, top_left, bottom_right, top_right

    @staticmethod
    def _rbf(d, gamma):
        """
        Computes the Radial Basis Function (RBF) for a given distance and gamma

        :param d: the Euclidean distance to a grid point
        :param gamma: gamma parameter
        :return: the RBF value for the distance d
        """
        return np.exp(-gamma * d ** 2)

    def rbf_interpolate(self):
        """
        Performs the Radial Basis function (RBF) interpolation for the point (x,y)

        :return: the interpolated value at (x,y)
        """

        x_minus, y_minus, i_minus, j_minus = self._get_coordinates()

        distances = self._euclidean_distances(x_minus, y_minus)

        h_diag_squared = self._hx ** 2 + self._hy ** 2
        gamma = -np.log(0.005) / h_diag_squared

        rbf_weights = [self._rbf(d, gamma) for d in distances]

        sum_rbf = np.sum(rbf_weights)
        interpolated = rbf_weights[0] * self._z[i_minus, j_minus]
        interpolated += rbf_weights[1] * self._z[i_minus, j_minus + 1]
        interpolated += rbf_weights[2] * self._z[i_minus + 1, j_minus]
        interpolated += rbf_weights[3] * self._z[i_minus + 1, j_minus + 1]
        interpolated /= sum_rbf

        return interpolated
