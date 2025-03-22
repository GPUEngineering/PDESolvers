import numpy as np
import matplotlib.pyplot as plt

from pdesolvers.enums.enums import OptionType

class MonteCarloPricing:

    def __init__(self, option_type: OptionType, S0, strike_price, r, sigma, T, time_steps, sim):
        """
        Initialize the Geometric Brownian Motion model with the given parameters.

        Parameters:
        - S0: Initial stock price
        - mu: Drift coefficient (expected return)
        - sigma: Volatility coefficient (standard deviation of returns)
        - T: Time period for the simulation (in years)
        - time_steps: Number of time steps in the simulation
        - sim: Number of simulations to run
        """

        self.__option_type = option_type
        self.__S0 = S0
        self.__strike_price = strike_price
        self.__r = r
        self.__sigma = sigma
        self.__T = T
        self.__time_steps = time_steps
        self.__sim = sim
        self.__S = None

    def get_monte_carlo_option_price(self):

        S = self.simulate_gbm()

        if self.__option_type == OptionType.EUROPEAN_CALL:
            payoff = np.maximum(S[:, -1] - self.__strike_price, 0)
        elif self.__option_type == OptionType.EUROPEAN_PUT:
            payoff = np.maximum(self.__strike_price - S[:, -1], 0)
        else:
            raise ValueError(f'Unsupported Option Type: {self.__option_type}')

        option_price = np.exp(-self.__r * self.__T) * np.mean(payoff)

        return option_price

    def simulate_gbm(self):
        """
        Simulate the Geometric Brownian Motion for the given parameters.

        This method calculates the stock prices at each time step for each simulation.
        """
        np.random.seed(42)
        t = self.__generate_grid()
        dt = t[1] - t[0]

        B = np.zeros((self.__sim, self.__time_steps))
        S = np.zeros((self.__sim, self.__time_steps))

        # for all simulations at t = 0
        S[:,0] = self.__S0
        Z = np.random.normal(0, 1, (self.__sim, self.__time_steps))

        for i in range(self.__sim):
            for j in range (1, self.__time_steps):
                # updates brownian motion
                B[i,j] = B[i,j-1] + np.sqrt(dt) * Z[i,j-1]
                # calculates stock price based on the incremental difference
                S[i,j] = S[i, j-1] * np.exp((self.__r - 0.5*self.__sigma**2)*dt + self.__sigma * (B[i, j] - B[i, j - 1]))

        self.__S = S

        return self.__S

    def __generate_grid(self):
        """
        Generate a time grid from 0 to T with `time_steps` intervals.

        Returns:
        - A numpy array representing the time grid.
        """

        return np.linspace(0, self.__T, self.__time_steps)

    def __get_stock_prices(self):
        """
        Get the simulated stock prices.

        Returns:
        - A numpy array of simulated stock prices.
        """

        return self.__S

    def plot(self):
        """
        Plot the simulated stock prices for all simulations.
        """

        t = self.__generate_grid()
        S = self.__get_stock_prices()

        fig = plt.figure(figsize=(10,6))
        for i in range(self.__sim):
            plt.plot(t, S[i])

        plt.title("Simulated Geometric Brownian Motion")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.show()