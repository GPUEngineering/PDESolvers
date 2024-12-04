import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

from pdesolvers.heat_equation_1d.solution.solution_1d import Solution1D
from pdesolvers.heat_equation_1d.solver.solver_strategy import SolverStrategy


class HeatEquationCNSolver(SolverStrategy):

    def __init__(self, equation):
        self.equation = equation

    def solve(self):

        x = self.equation.generate_x_grid()
        t = self.equation.generate_t_grid()

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        alpha = self.equation.get_k() * dt / (2 * dx**2)
        a = -alpha
        b = 1 + 2 * alpha
        c = -alpha

        u = np.zeros((self.equation.get_t_nodes(), self.equation.get_x_nodes()))

        u[0, :] = self.equation.get_initial_temp(x)
        u[:, 0] = self.equation.get_left_boundary(t)
        u[:, -1] = self.equation.get_right_boundary(t)

        lhs = self.__build_tridiagonal_matrix(a, b, c, self.equation.get_x_nodes() - 2)
        rhs = np.zeros(self.equation.get_x_nodes() - 2)

        for tau in range(0, self.equation.get_t_nodes() - 1):
            rhs[0] = alpha * (u[tau, 0] + u[tau+1, 0]) + (1 - 2 * alpha) * u[tau, 1] + alpha * u[tau, 2]

            for i in range(1, self.equation.get_x_nodes() - 2):
                rhs[i] = alpha * u[tau, i] + (1 - 2 * alpha) * u[tau, i+1] + alpha * u[tau, i+2]

            rhs[-1] = alpha * (u[tau, -1] + u[tau+1, -1]) + (1 - 2 * alpha) * u[tau, -2] + alpha * u[tau, -3]

            u[tau+1, 1:-1] = spsolve(lhs, rhs)

        return Solution1D(u, x, t)

    @staticmethod
    def __build_tridiagonal_matrix(a, b, c, nodes):
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
