import numpy as np

class BlackScholesEquation:

    def __init__(self, option_type, S_max, expiry, sigma, r, K, s_nodes=1, t_nodes=None):
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

    @property
    def s_nodes(self):
        return self.__s_nodes

    @property
    def t_nodes(self):
        return self.__t_nodes

    @property
    def option_type(self):
        return self.__option_type

    @property
    def S_max(self):
        return self.__S_max

    @property
    def sigma(self):
        return self.__sigma

    @property
    def expiry(self):
        return self.__expiry

    @property
    def rate(self):
        return self.__r

    @property
    def strike_price(self):
        return self.__K

    @t_nodes.setter
    def t_nodes(self, nodes):
        self.__t_nodes = nodes

    @option_type.setter
    def option_type(self, type):
        self.__option_type = type

    @S_max.setter
    def S_max(self, asset_price):
        self.__S_max = asset_price

    @expiry.setter
    def expiry(self, expiry):
        self.__expiry = expiry

    @sigma.setter
    def sigma(self, sigma):
        self.__sigma = sigma

    @rate.setter
    def rate(self, rate):
        self.__r = rate

    @strike_price.setter
    def strike_price(self, price):
        self.__K = price