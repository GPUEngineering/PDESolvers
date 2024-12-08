import numpy as np
import matplotlib.pyplot as plt

class GeometricBrownianMotion:

    def __init__(self, S0, mu, sigma, T, time_steps, sim):
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

        self.__S0 = S0
        self.__mu = mu
        self.__sigma = sigma
        self.__T = T
        self.__time_steps = time_steps
        self.__sim = sim
        self.__S = None

    def simulate_gbm(self):
        """
        Simulate the Geometric Brownian Motion for the given parameters.

        This method calculates the stock prices at each time step for each simulation.
        """

        t = self.__generate_grid()
        dt = t[1] - t[0]


        S = np.zeros((self.__sim, self.__time_steps))

        # for all simulations at t = 0
        S[:,0] = self.__S0

        for i in range(self.__sim):
            for j in range (1, self.__time_steps):
                # calculates stock price based on the previous value at j-1
                S[i,j] = S[i, j-1] * np.exp((self.__mu - 0.5*self.__sigma**2)*dt + self.__sigma*np.sqrt(dt)*(np.random.normal(0, 1)))

        self.__S = S

        return self

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


def main():
    GeometricBrownianMotion(100, 0.05, 0.03, 1, 365, 10).simulate_gbm().plot()

if __name__ == "__main__":
    main()

