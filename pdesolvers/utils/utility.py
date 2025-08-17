import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse import csc_matrix
import cupy as cp
from cupyx.scipy.sparse import diags as cupy_diags

import pdesolvers.enums.enums as enum

class Heat1DHelper:

    @staticmethod
    def _build_tridiagonal_matrix(a, b, c, nodes):
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

class Heat2DHelper:
    @staticmethod
    def initMatrix(t_nodes, x_nodes, y_nodes, left, right, bottom, top, u0, xDomain, yDomain, tDomain):
        """
        Initialize a 3D matrix with boundary and initial conditions for the 2D heat equation.
        
        Parameters:
        -----------
        t_nodes : int
            Number of time points
        x_nodes : int  
            Number of x spatial points
        y_nodes : int
            Number of y spatial points
        left : callable
            Left boundary condition function: left(t, y) -> temperature
        right : callable  
            Right boundary condition function: right(t, y) -> temperature
        bottom : callable
            Bottom boundary condition function: bottom(t, x) -> temperature
        top : callable
            Top boundary condition function: top(t, x) -> temperature
        u0 : callable
            Initial condition function: u0(x, y) -> temperature
        xDomain : np.array
            X coordinate array
        yDomain : np.array  
            Y coordinate array
        tDomain : np.array
            Time array
            
        Returns:
        --------
        np.array
            3D array of shape (t_nodes, x_nodes, y_nodes) with initialized conditions
        """
        
        matrix = np.zeros((t_nodes, x_nodes, y_nodes))
        
        for tau in range(t_nodes):
            t = tDomain[tau]
            
            # Left and right boundaries (x = 0 and x = xLength)
            for i in range(y_nodes):
                y = yDomain[i]
                matrix[tau, 0, i] = left(t, y)      # Left boundary (x=0)
                matrix[tau, -1, i] = right(t, y)    # Right boundary (x=xLength)
            
            # Bottom and top boundaries (y = 0 and y = yLength)  
            for j in range(x_nodes):
                x = xDomain[j]
                matrix[tau, j, 0] = bottom(t, x)    # Bottom boundary (y=0)
                matrix[tau, j, -1] = top(t, x)      # Top boundary (y=yLength)
        
        # Set initial condition at t=0
        for i in range(x_nodes):
            for j in range(y_nodes):
                try:
                    initial_val = u0(xDomain[i], yDomain[j])
                    if hasattr(initial_val, '__iter__') and not isinstance(initial_val, str):
                        matrix[0, i, j] = float(initial_val.flat[0]) if hasattr(initial_val, 'flat') else float(initial_val[0])
                    else:
                        matrix[0, i, j] = float(initial_val)
                except (TypeError, IndexError, AttributeError):
                    matrix[0, i, j] = float(u0(xDomain[i], yDomain[j]))
        
        for tau in range(t_nodes):
            t = tDomain[tau]
            matrix[tau, 0, 0] = left(t, yDomain[0])      # Bottom-left
            matrix[tau, 0, -1] = left(t, yDomain[-1])    # Top-left  
            matrix[tau, -1, 0] = right(t, yDomain[0])    # Bottom-right
            matrix[tau, -1, -1] = right(t, yDomain[-1])  # Top-right
        
        return matrix

    def innitTriDiagMatrix(nx, ny, cx, cy, alpha):
        """
        Build sparse matrix G for: -cx*U[i-1,j] + α*U[i,j] - cx*U[i+1,j] - cy*U[i,j-1] - cy*U[i,j+1] = RHS
        """
        n_interior_x = nx - 2
        n_interior_y = ny - 2
        n_total = n_interior_x * n_interior_y
        
        # Main diagonal: α coefficients
        main_diagonal = np.full(n_total, alpha)
        
        # x-direction coupling: -cx coefficients (±1 in flattened index)
        x_off_diagonal = np.full(n_total - 1, -cx)
        # Zero out connections across y-boundaries
        for i in range(n_interior_x - 1, n_total - 1, n_interior_x):
            if i < len(x_off_diagonal):
                x_off_diagonal[i] = 0
        
        # y-direction coupling: -cy coefficients (±n_interior_x in flattened index)
        y_off_diagonal = np.full(n_total - n_interior_x, -cy)
        
        # Assemble sparse matrix
        diagonals = [y_off_diagonal, x_off_diagonal, main_diagonal, 
                    x_off_diagonal[:n_total-1], y_off_diagonal]
        offsets = [-n_interior_x, -1, 0, 1, n_interior_x]
        
        G = diags(diagonals, offsets=offsets, shape=(n_total, n_total), format='csr')
        
        return G, n_interior_x, n_interior_y
    
    @staticmethod
    def init_matrix_gpu(t_nodes, x_nodes, y_nodes, left, right, bottom, top, u0, x_domain, y_domain, t_domain):
        """
        GPU version of initMatrix - initializes on GPU then returns CuPy array
        
        Parameters same as initMatrix but returns CuPy array for GPU computation
        """
        # First initialize on CPU using existing method
        matrix_cpu = Heat2DHelper.initMatrix(t_nodes, x_nodes, y_nodes, left, right, bottom, top, u0, x_domain, y_domain, t_domain)
        
        # Transfer to GPU
        matrix_gpu = cp.asarray(matrix_cpu)
        
        return matrix_gpu

    @staticmethod
    def build_cupy_sparse_matrix(nx, ny, cx, cy, alpha):
        """
        Build sparse matrix for Crank-Nicolson system using CuPy
        
        Parameters:
        -----------
        nx, ny : int
            Number of spatial nodes in x and y directions
        cx, cy : float  
            Stability parameters in x and y directions
        alpha : float
            Main diagonal coefficient
            
        Returns:
        --------
        G : cupyx.scipy.sparse matrix
            System matrix for implicit solve
        n_interior_x, n_interior_y : int
            Number of interior nodes in each direction
        """
        n_interior_x = nx - 2
        n_interior_y = ny - 2
        n_total = n_interior_x * n_interior_y
        
        # Main diagonal: α coefficients
        main_diagonal = cp.full(n_total, alpha)
        
        # x-direction coupling: -cx coefficients (±1 in flattened index)
        x_off_diagonal = cp.full(n_total - 1, -cx)
        # Zero out connections across y-boundaries
        for i in range(n_interior_x - 1, n_total - 1, n_interior_x):
            if i < len(x_off_diagonal):
                x_off_diagonal[i] = 0
        
        # y-direction coupling: -cy coefficients (±n_interior_x in flattened index)
        y_off_diagonal = cp.full(n_total - n_interior_x, -cy)
        
        # Assemble sparse matrix using CuPy
        diagonals = [y_off_diagonal, x_off_diagonal, main_diagonal, 
                    x_off_diagonal[:n_total-1], y_off_diagonal]
        offsets = [-n_interior_x, -1, 0, 1, n_interior_x]
        
        G = cupy_diags(diagonals, offsets=offsets, shape=(n_total, n_total), format='csr')
        
        return G, n_interior_x, n_interior_y
    
class BlackScholesHelper:

    @staticmethod
    def _calculate_greeks_at_boundary(equation, delta, gamma, theta, tau, V, S, ds):
        delta[0, tau] = (V[1, tau+1] - V[0, tau+1]) / ds
        delta[equation.s_nodes, tau] = (V[equation.s_nodes, tau+1] - V[equation.s_nodes-1, tau+1]) / ds

        gamma[0, tau] = (V[2, tau+1] - 2*V[1, tau+1] + V[0, tau+1]) / (ds**2)
        gamma[equation.s_nodes, tau] = (V[equation.s_nodes, tau+1] - 2*V[equation.s_nodes-1, tau+1] + V[equation.s_nodes-2, tau+1]) / (ds**2)

        theta[0, tau] = -0.5 * (equation.sigma**2) * (S[0]**2) * gamma[0, tau] - equation.rate * S[0] * delta[0, tau] + equation.rate * V[0, tau+1]
        theta[equation.s_nodes, tau] = -0.5 * (equation.sigma**2) * (S[-1]**2) * gamma[equation.s_nodes, tau] - equation.rate * S[-1] * delta[equation.s_nodes, tau] + equation.rate * V[equation.s_nodes, tau+1]

        return delta, gamma, theta

    @staticmethod
    def _set_boundary_conditions(equation, T, tau):
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

class GPUResults:

    def __init__(self, file_path, s_max, expiry):
        self.__file_path = file_path
        self.__s_max = s_max
        self.__expiry = expiry
        self.__grid_data = None

    def get_results(self):

        # Load data
        df = pd.read_csv(self.__file_path, header=None)
        print(f"Data shape: {df.shape}")

        self.__grid_data = df.values.T

        return self.__grid_data

    def plot_option_surface(self):
        if self.__grid_data is None:
            self.get_results()

        price_grid = np.linspace(0, self.__s_max, self.__grid_data.shape[0])
        time_grid = np.linspace(0, self.__expiry, self.__grid_data.shape[1])
        X, Y = np.meshgrid(time_grid, price_grid)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self.__grid_data, cmap='viridis')

        ax.set_xlabel('Time')
        ax.set_ylabel('Asset Price')
        ax.set_zlabel('Option Value')
        ax.set_title('Option Value Surface Plot')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()