import numpy as np
from scipy.stats import norm
from pdesolvers.enums.enums import OptionType

class BlackScholesFormula:
    def __init__(self, option_type: OptionType, S0, strike_price, r, sigma, expiry):
        self.__option_type = option_type
        self.__S0 = S0
        self.__strike_price = strike_price
        self.__r = r
        self.__sigma = sigma
        self.__expiry = expiry

    def get_black_scholes_merton_price(self):

        d1 = (np.log(self.__S0/self.__strike_price) + (self.__r + 0.5 * self.__sigma**2) * self.__expiry) / (self.__sigma * np.sqrt(self.__expiry))
        d2 = d1 - (self.__sigma * np.sqrt(self.__expiry))

        if self.__option_type == OptionType.EUROPEAN_CALL:
            payoff = self.__S0 * norm.cdf(d1) - self.__strike_price * np.exp(-self.__r * self.__expiry) * norm.cdf(d2)
        elif self.__option_type == OptionType.EUROPEAN_PUT:
            payoff =  self.__strike_price * np.exp(-self.__r * self.__expiry) * norm.cdf(-d2) - self.__S0 * norm.cdf(-d1)
        else:
            raise ValueError(f'Unsupported option type: {self.__option_type}')

        return payoff