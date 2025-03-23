import numpy as np
import pdesolvers.solution as sol
import pdesolvers.pdes.black_scholes as bse
import pdesolvers.enums.enums as enum
import pdesolvers.utils.utility as utility

from scipy import sparse
from scipy.sparse.linalg import spsolve

class BlackScholesExplicitSolver:

    def __init__(self, equation: bse.BlackScholesEquation):
        self.equation = equation

    def solve(self):
        """
        This method solves the Black-Scholes equation using the explicit finite difference method

        :return: the solver instance with the computed option values
        """

        S = self.equation.generate_grid(self.equation.S_max, self.equation.s_nodes)
        T = self.equation.generate_grid(self.equation.expiry, self.equation.t_nodes)

        dt_max = 1/((self.equation.s_nodes**2) * (self.equation.sigma**2)) # cfl condition to ensure stability

        if self.equation.t_nodes is None:
            dt = 0.9 * dt_max
            self.equation.t_nodes = int(self.equation.expiry/dt)
            dt = self.equation.expiry / self.equation.t_nodes
        else:
            dt = T[1] - T[0]

            if dt > dt_max:
                raise ValueError("User-defined t nodes is too small and exceeds the CFL condition. Possible action: Increase number of t nodes for stability!")

        ds = S[1] - S[0]

        V = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))

        # setting terminal condition
        if self.equation.option_type == enum.OptionType.EUROPEAN_CALL:
            V[:,-1] = np.maximum((S - self.equation.strike_price), 0)
        elif self.equation.option_type == enum.OptionType.EUROPEAN_PUT:
            V[:,-1] = np.maximum((self.equation.strike_price - S), 0)
        else:
            raise ValueError("Invalid option type - please choose between call/put")

        delta = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))
        gamma = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))
        theta = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))

        for tau in reversed(range(self.equation.t_nodes)):
            for i in range(1, self.equation.s_nodes):
                delta[i, tau] = (V[i+1, tau+1] - V[i-1, tau+1]) / (2 * ds)
                gamma[i, tau] = (V[i+1, tau+1] - 2 * V[i,tau+1] + V[i-1, tau+1]) / (ds ** 2)
                theta[i, tau] = -0.5 * (self.equation.sigma ** 2) * (S[i] ** 2) * gamma[i, tau] - self.equation.rate * S[i] * delta[i, tau] + self.equation.rate * V[i, tau+1]
                V[i, tau] = V[i, tau + 1] - (theta[i, tau] * dt)

            # setting boundary conditions
            lower, upper = utility.BlackScholesHelper.set_boundary_conditions(self.equation, T, tau)
            V[0, tau] = lower
            V[self.equation.s_nodes, tau] = upper

            delta, gamma, theta = utility.BlackScholesHelper.calculate_greeks_at_boundary(self.equation, delta, gamma, theta, tau, V, S, ds)

        return sol.SolutionBlackScholes(V, T, S, dt, ds, delta, gamma, theta, self.equation.option_type)


class BlackScholesCNSolver:

    def __init__(self, equation: bse.BlackScholesEquation):
        self.equation = equation

    def solve(self):
        """
        This method solves the Black-Scholes equation using the Crank-Nicolson method

        :return: the solver instance with the computed option values
        """

        S = self.equation.generate_grid(self.equation.S_max, self.equation.s_nodes)
        T = self.equation.generate_grid(self.equation.expiry, self.equation.t_nodes)

        ds = S[1] - S[0]
        dt = T[1] - T[0]

        a = 0.25 * dt * ((self.equation.sigma**2) * (S**2) / (ds**2) - self.equation.rate * S / ds)
        b = -dt * 0.5 * (self.equation.sigma**2 * (S**2) / (ds**2) + self.equation.rate)
        c = 0.25 * dt * (self.equation.sigma**2 * (S**2) / (ds**2) + self.equation.rate * S / ds)

        lhs = sparse.diags([-a[2:], 1-b[1:], -c[1:-1]], [-1, 0, 1], shape = (self.equation.s_nodes - 1, self.equation.s_nodes - 1), format='csr')
        rhs = sparse.diags([a[2:], 1+b[1:], c[1:-1]], [-1, 0, 1], shape = (self.equation.s_nodes - 1, self.equation.s_nodes - 1) , format='csr')

        V = np.zeros((self.equation.s_nodes+1, self.equation.t_nodes+1))

        delta = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))
        gamma = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))
        theta = np.zeros((self.equation.s_nodes + 1, self.equation.t_nodes + 1))


# setting terminal condition (for all values of S at time T)
        if self.equation.option_type == enum.OptionType.EUROPEAN_CALL:
            V[:,-1] = np.maximum((S - self.equation.strike_price), 0)

            # setting boundary conditions (for all values of t at asset prices S=0 and S=Smax)
            V[0, :] = 0
            V[-1, :] = S[-1] - self.equation.strike_price * np.exp(-self.equation.rate * (self.equation.expiry - T))

        elif self.equation.option_type == enum.OptionType.EUROPEAN_PUT:
            V[:,-1] = np.maximum((self.equation.strike_price - S), 0)
            V[0, :] = self.equation.strike_price * np.exp(-self.equation.rate * (self.equation.expiry - T))
            V[-1, :] = 0

        for tau in reversed(range(self.equation.t_nodes)):
            # Construct the RHS vector for this time step
            rhs_vector = rhs @ V[1:-1, tau + 1]

            # Apply boundary conditions to the RHS vector
            rhs_vector[0] += a[1] * (V[0, tau + 1] + V[0, tau])
            rhs_vector[-1] += c[self.equation.s_nodes-1] *(V[-1, tau+1] + V[-1, tau])

            # Solve the linear system for interior points
            V[1:-1, tau] = spsolve(lhs, rhs_vector)

            # Calculate Greeks for interior points
            delta[1:-1, tau] = (V[2:, tau] - V[:-2, tau]) / (2 * ds)
            gamma[1:-1, tau] = (V[2:, tau] - 2 * V[1:-1, tau] + V[:-2, tau]) / (ds**2)
            theta[1:-1, tau] = -0.5 * (self.equation.sigma**2) * (S[1:-1]**2) * gamma[1:-1, tau] - self.equation.rate * S[1:-1] * delta[1:-1, tau] + self.equation.rate * V[1:-1, tau]

            delta, gamma, theta = utility.BlackScholesHelper.calculate_greeks_at_boundary(self.equation, delta, gamma, theta, tau, V, S, ds)

        return sol.SolutionBlackScholes(V, T, S, dt, ds, delta, gamma, theta, self.equation.option_type)

