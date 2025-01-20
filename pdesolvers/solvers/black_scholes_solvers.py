from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pdesolvers.solution as sol

class BlackScholesExplicitSolver:

    def __init__(self, equation):
        self.equation = equation

    def solve(self):
        """
        This method solves the Black-Scholes equation using the explicit finite difference method
        :return: the solver instance with the computed option values
        """

        if self.equation.get_t_nodes() is None:
            dt_max = 1/((self.equation.get_s_nodes()**2) * (self.equation.get_sigma()**2)) # cfl condition to ensure stability
            dt = 0.9 * dt_max
            self.equation.set_t_nodes(int(self.equation.get_expiry()/dt))
            dt = self.equation.get_expiry() / self.equation.get_t_nodes() # to ensure that the expiration time is integer time steps away
        else:
            # possible fix - set a check to see that user-defined value is within cfl condition
            dt = self.equation.get_expiry() / self.equation.get_t_nodes()

        S = self.equation.generate_asset_grid()
        T = self.equation.generate_time_grid()

        dS = S[1] - S[0]

        V = np.zeros((self.equation.get_s_nodes() + 1, self.equation.get_t_nodes() + 1))

        # setting terminal condition
        if self.equation.get_option_type() == 'call':
            V[:,-1] = np.maximum((S - self.equation.get_strike_price()), 0)
        elif self.equation.get_option_type() == 'put':
            V[:,-1] = np.maximum((self.equation.get_strike_price() - S), 0)
        else:
            raise ValueError("Invalid option type - please choose between call/put")

        for tau in reversed(range(self.equation.get_t_nodes())):
            for i in range(1, self.equation.get_s_nodes()):
                delta = (V[i+1, tau+1] - V[i-1, tau+1]) /  (2 * dS)
                gamma = (V[i+1, tau+1] - 2 * V[i,tau+1] + V[i-1, tau+1]) / dS ** 2
                theta = -0.5 * ( self.equation.get_sigma() ** 2) * (S[i] ** 2) * gamma - self.equation.get_rate() * S[i] * delta + self.equation.get_rate() * V[i, tau+1]
                V[i, tau] = V[i, tau + 1] - (theta * dt)

            # setting boundary conditions
            lower, upper = self.__set_boundary_conditions(T, tau)
            V[0, tau] = lower
            V[self.equation.get_s_nodes(), tau] = upper


        return sol.SolutionBlackScholes(V,S,T)

    def __set_boundary_conditions(self, T, tau):
        lower_boundary = None
        upper_boundary = None
        if self.equation.get_option_type() == 'call':
            lower_boundary = 0
            upper_boundary = self.equation.get_S_max() - self.equation.get_strike_price() * np.exp(-self.equation.get_rate() * (self.equation.get_expiry() - T[tau]))
        elif self.equation.get_option_type() == 'put':
            lower_boundary = self.equation.get_strike_price() * np.exp(-self.equation.get_rate() * (self.equation.get_expiry() - T[tau]))
            upper_boundary = 0

        return lower_boundary, upper_boundary

class BlackScholesCNSolver:

    def __init__(self, equation):
        self.equation = equation

    def solve(self):

        S = self.equation.generate_asset_grid()
        T = self.equation.generate_time_grid()

        dS = S[1] - S[0]
        dT = T[1] - T[0]

        alpha = 0.25 * dT * ((self.equation.get_sigma()**2) * (S**2) - self.equation.get_rate()*S)
        beta = -dT * 0.5 * (self.equation.get_sigma()**2 * (S**2) + self.equation.get_rate())
        gamma = 0.25 * dT * (self.equation.get_sigma()**2 * (S**2) + self.equation.get_rate() * S)

        lhs = sparse.diags([-alpha[2:], 1-beta[1:], -gamma[1:-1]], [-1, 0, 1], shape = (self.equation.get_s_nodes() - 1, self.equation.get_s_nodes() - 1), format='csr')
        rhs = sparse.diags([alpha[2:], 1+beta[1:], gamma[1:-1]], [-1, 0, 1], shape = (self.equation.get_s_nodes() - 1, self.equation.get_s_nodes() - 1) , format='csr')

        V = np.zeros((self.equation.get_s_nodes()+1, self.equation.get_t_nodes()+1))

        # setting terminal condition (for all values of S at time T)
        if self.equation.get_option_type() == 'call':
            V[:,-1] = np.maximum((S - self.equation.get_strike_price()), 0)

            # setting boundary conditions (for all values of t at asset prices S=0 and S=Smax)
            V[0, :] = 0
            V[-1, :] = S[-1] - self.equation.get_strike_price() * np.exp(-self.equation.get_rate() * (self.equation.get_expiry() - T))

        elif self.equation.get_option_type() == 'put':
            V[:,-1] = np.maximum((self.equation.get_strike_price() - S), 0)
            V[0, :] = self.equation.get_strike_price() * np.exp(-self.equation.get_rate() * (self.equation.get_expiry() - T))
            V[-1, :] = 0

        for tau in reversed(range(self.equation.get_t_nodes())):
            # Construct the RHS vector for this time step
            rhs_vector = rhs @ V[1:-1, tau + 1]

            # Apply boundary conditions to the RHS vector
            rhs_vector[0] += alpha[1] * (V[0, tau + 1] + V[0, tau])
            rhs_vector[-1] += gamma[self.equation.get_s_nodes()-1] *(V[-1, tau+1] + V[-1, tau])

            # Solve the linear system for interior points
            V[1:-1, tau] = spsolve(lhs, rhs_vector)

        return sol.SolutionBlackScholes(V,S,T)
