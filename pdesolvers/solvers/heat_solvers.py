import numpy as np
import pdesolvers.solution as sol
import pdesolvers.pdes.heat_1d as heat
import pdesolvers.utils.utility as utility

from scipy.sparse.linalg import spsolve

class Heat1DExplicitSolver:
    def __init__(self, equation: heat.HeatEquation):
        self.equation = equation

    def solve(self):
        """
        This method solves the heat (diffusion) equation using the explicit finite difference method

        :return: the solver instance with the computed temperature values
        """

        x = self.equation.generate_grid(self.equation.length, self.equation.x_nodes)
        t = self.equation.generate_grid(self.equation.time, self.equation.t_nodes)

        dx = x[1] - x[0]
        dt_max = 0.5 * (dx**2) / self.equation.k

        if self.equation.t_nodes is None:
            dt = 0.8 * dt_max
            self.equation.t_nodes = int(self.equation.time/dt)
            dt = self.equation.time / self.equation.t_nodes
        else:
            dt = t[1] - t[0]

        if dt > dt_max:
            raise ValueError("User-defined t nodes is too small and exceeds the CFL condition. Possible action: Increase number of t nodes for stability!")

        u = np.zeros((self.equation.t_nodes, self.equation.x_nodes))

        u[0, :] = self.equation.get_initial_temp(x)
        u[:, 0] = self.equation.get_left_boundary(t)
        u[:, -1] = self.equation.get_right_boundary(t)

        for tau in range(0, self.equation.t_nodes-1):
            for i in range(1, self.equation.x_nodes - 1):
                u[tau+1,i] = u[tau, i] + (dt * self.equation.k * (u[tau, i-1] - 2 * u[tau, i] + u[tau, i+1]) / dx**2)

        return sol.Solution1D(u, x, t, dx, dt)

class Heat1DCNSolver:
    def __init__(self, equation: heat.HeatEquation):
        self.equation = equation

    def solve(self):
        """
        This method solves the heat (diffusion) equation using the Crank Nicolson method

        :return: the solver instance with the computed temperature values
        """

        x = self.equation.generate_grid(self.equation.length, self.equation.x_nodes)
        t = self.equation.generate_grid(self.equation.time, self.equation.t_nodes)

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        alpha = self.equation.k * dt / (2 * dx**2)
        a = -alpha
        b = 1 + 2 * alpha
        c = -alpha

        u = np.zeros((self.equation.t_nodes, self.equation.x_nodes))

        u[0, :] = self.equation.get_initial_temp(x)
        u[:, 0] = self.equation.get_left_boundary(t)
        u[:, -1] = self.equation.get_right_boundary(t)

        lhs = utility.Heat1DHelper.build_tridiagonal_matrix(a, b, c, self.equation.x_nodes - 2)
        rhs = np.zeros(self.equation.x_nodes - 2)

        for tau in range(0, self.equation.t_nodes - 1):
            rhs[0] = alpha * (u[tau, 0] + u[tau+1, 0]) + (1 - 2 * alpha) * u[tau, 1] + alpha * u[tau, 2]

            for i in range(1, self.equation.x_nodes - 2):
                rhs[i] = alpha * u[tau, i] + (1 - 2 * alpha) * u[tau, i+1] + alpha * u[tau, i+2]

            rhs[-1] = alpha * (u[tau, -1] + u[tau+1, -1]) + (1 - 2 * alpha) * u[tau, -2] + alpha * u[tau, -3]

            u[tau+1, 1:-1] = spsolve(lhs, rhs)

        return sol.Solution1D(u, x, t, dx, dt)