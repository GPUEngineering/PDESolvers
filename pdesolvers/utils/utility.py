import numpy as np
from scipy.sparse import csc_matrix

import pdesolvers.enums.enums as enum

class Heat1DHelper:

    @staticmethod
    def build_tridiagonal_matrix(a, b, c, nodes):
        """
        Initialises the tridiagonal matrix on the LHS of the equation

        :param a: the coefficient of U @ (t = tau + 1 & x = i-1)
        :param b: the coefficient of U @ (t = tau + 1 & x = i)
        :param c: the coefficient of U @ (t = tau + 1 & x = i+1)
        :param nodes: number of spatial nodes ( used to initialise the size of the tridiagonal matrix)
        :return: the tridiagonal matrix consisting of coefficients
        """

        matrix = np.zeros((nodes, nodes))
        np.fill_diagonal(matrix, b)
        np.fill_diagonal(matrix[1:], a)
        np.fill_diagonal(matrix[:, 1:], c)

        matrix = csc_matrix(matrix)

        return matrix

class BlackScholesHelper:

    @staticmethod
    def calculate_greeks_at_boundary(equation, delta, gamma, theta, tau, V, S, ds):
        delta[0, tau] = (V[1, tau+1] - V[0, tau+1]) / ds
        delta[equation.s_nodes, tau] = (V[equation.s_nodes, tau+1] - V[equation.s_nodes-1, tau+1]) / ds

        gamma[0, tau] = (V[2, tau+1] - 2*V[1, tau+1] + V[0, tau+1]) / (ds**2)
        gamma[equation.s_nodes, tau] = (V[equation.s_nodes, tau+1] - 2*V[equation.s_nodes-1, tau+1] + V[equation.s_nodes-2, tau+1]) / (ds**2)

        theta[0, tau] = -0.5 * (equation.sigma**2) * (S[0]**2) * gamma[0, tau] - equation.rate * S[0] * delta[0, tau] + equation.rate * V[0, tau+1]
        theta[equation.s_nodes, tau] = -0.5 * (equation.sigma**2) * (S[-1]**2) * gamma[equation.s_nodes, tau] - equation.rate * S[-1] * delta[equation.s_nodes, tau] + equation.rate * V[equation.s_nodes, tau+1]

        return delta, gamma, theta

    @staticmethod
    def set_boundary_conditions(equation, T, tau):
        """
        Sets the boundary conditions for the Black-Scholes Equation based on option type

        :param T: grid of time steps
        :param tau: index of current time step
        :return: a tuple representing the boundary values for the given time step
        """

        lower_boundary = None
        upper_boundary = None
        if equation.option_type == enum.OptionType.EUROPEAN_CALL:
            lower_boundary = 0
            upper_boundary = equation.S_max - equation.strike_price * np.exp(-equation.rate * (equation.expiry - T[tau]))
        elif equation.option_type == enum.OptionType.EUROPEAN_PUT:
            lower_boundary = equation.strike_price * np.exp(-equation.rate * (equation.expiry - T[tau]))
            upper_boundary = 0

        return lower_boundary, upper_boundary

class RBFInterpolator:

    def __init__(self, z, hx, hy):
        """
        Initializes the RBF Interpolator.

        :param z: 2D array of values at the grid points.
        :param x: x-coordinate of the point to interpolate.
        :param y: y-coordinate of the point to interpolate.
        :param hx: Grid spacing in the x-direction.
        :param hy: Grid spacing in the y-direction.
        """

        self.__z = z
        self.__hx = hx
        self.__hy = hy
        self.__nx, self.__ny = z.shape

    def __get_coordinates(self, x, y):
        """
        Determines the x and y coordinates of the bottom-left corner of the grid cell

        :return: A tuple containing the coordinates and its corresponding indices
        """

        # gets the grid steps to x
        i_minus_star = int(np.floor(x / self.__hx))
        i_minus = min(max(0, i_minus_star), self.__nx - 2)

        # gets the grid steps to y
        j_minus_star = int(np.floor(y / self.__hy))
        j_minus = min(max(0, j_minus_star), self.__ny - 2)

        # computes the coordinates at the computed indices
        x_minus = i_minus * self.__hx
        y_minus = j_minus * self.__hy

        return x_minus, y_minus, i_minus, j_minus

    def __euclidean_distances(self, x_minus, y_minus, x, y):
        """
        Calculates Euclidean distances between (x,y) and the surrounding grid points in the unit cell

        :param x_minus: x-coordinate of the bottom-left corner of the grid
        :param y_minus: y-coordinate of the bottom-left corner of the grid
        :return: returns tuple with the Euclidean distances to the surrounding grid points:
                [bottom left, top left, bottom right, top right]
        """

        bottom_left = np.sqrt((x_minus - x) ** 2 + (y_minus - y) ** 2)
        top_left = np.sqrt((x_minus - x) ** 2 + (y_minus + self.__hy - y) ** 2)
        bottom_right = np.sqrt((x_minus + self.__hx - x) ** 2 + (y_minus - y) ** 2)
        top_right = np.sqrt((x_minus + self.__hx - x) ** 2 + (y_minus + self.__hy - y) ** 2)

        return bottom_left, top_left, bottom_right, top_right

    @staticmethod
    def __rbf(d, gamma):
        """
        Computes the Radial Basis Function (RBF) for a given distance and gamma

        :param d: the Euclidean distance to a grid point
        :param gamma: gamma parameter
        :return: the RBF value for the distance d
        """
        return np.exp(-gamma * d ** 2)

    def interpolate(self, x, y):
        """
        Performs the Radial Basis function (RBF) interpolation for the point (x,y)

        :return: the interpolated value at (x,y)
        """

        x_minus, y_minus, i_minus, j_minus = self.__get_coordinates(x, y)

        distances = self.__euclidean_distances(x_minus, y_minus, x, y)

        h_diag_squared = self.__hx ** 2 + self.__hy ** 2
        gamma = -np.log(0.005) / h_diag_squared

        rbf_weights = [self.__rbf(d, gamma) for d in distances]

        sum_rbf = np.sum(rbf_weights)
        interpolated = rbf_weights[0] * self.__z[i_minus, j_minus]
        interpolated += rbf_weights[1] * self.__z[i_minus, j_minus + 1]
        interpolated += rbf_weights[2] * self.__z[i_minus + 1, j_minus]
        interpolated += rbf_weights[3] * self.__z[i_minus + 1, j_minus + 1]
        interpolated /= sum_rbf

        return interpolated