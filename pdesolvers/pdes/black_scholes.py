import numpy as np

class BlackScholesEquation:

    def __init__(self, option_type, S_max, expiry, sigma, r, K, s_nodes, t_nodes = None):
        """
        Initialises the solver with the necessary parameters

        :param option_type: the type of option
        :param S_max: maximum asset price in the grid
        :param expiry: time to maturity/expiry of the option
        :param sigma: volatility of the asset
        :param r: risk-free interest rate
        :param K: strike price
        :param s_nodes: number of asset price nodes
        :param t_nodes: number of time nodes
        """

        self.__option_type = option_type
        self.__S_max = S_max
        self.__expiry = expiry
        self.__sigma = sigma
        self.__r = r
        self.__K = K
        self.__s_nodes = s_nodes
        self.__t_nodes = t_nodes
        self.__V = None

    def generate_asset_grid(self):
        return np.linspace(0, self.__S_max, self.__s_nodes+1)

    def generate_time_grid(self):
        return np.linspace(0, self.__expiry, self.__t_nodes+1)

    def set_t_nodes(self, nodes):
        self.__t_nodes = nodes

    def set_option_type(self, type):
        self.__option_type = type

    def set_S_max(self, asset_price):
        self.__S_max = asset_price

    def set_expiry(self, expiry):
        self.__expiry = expiry

    def set_sigma(self, sigma):
        self.__sigma = sigma

    def set_rate(self, rate):
        self.__r = rate

    def set_strike_price(self, price):
        self.__K = price

    def get_s_nodes(self):
        return self.__s_nodes

    def get_t_nodes(self):
        return self.__t_nodes

    def get_option_type(self):
        return self.__option_type

    def get_S_max(self):
        return self.__S_max

    def get_sigma(self):
        return self.__sigma

    def get_expiry(self):
        return self.__expiry

    def get_rate(self):
        return self.__r

    def get_strike_price(self):
        return self.__K