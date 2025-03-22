import numpy as np
import yfinance as yf

from pdesolvers.enums.enums import OptionType

class HistoricalStockData:
    def __init__(self, ticker):
        self.__stock_data = None
        self.__options = None
        self.__ticker = ticker

    def fetch_stock_data(self, start_date, end_date):
        self.__stock_data = yf.download(self.__ticker, start=start_date, end=end_date, interval='1d')
        return self.__stock_data

    def estimate_metrics(self):
        if self.__stock_data is None:
            print(f'No data available - data must be fetched first.')

        self.__stock_data["Log Returns"] = np.log(self.__stock_data["Close"] / self.__stock_data["Close"].shift(1))
        log_returns = self.__stock_data["Log Returns"].dropna()

        sigma = log_returns.std()
        mu = log_returns.mean()

        return sigma, mu

    def get_latest_stock_price(self):
        if self.__stock_data is None:
            raise ValueError("No data available. Call fetch_data first.")

        return self.__stock_data["Close"].iloc[-1].item()

class OptionsData:

    def __init__(self, ticker):
        self.__ticker = ticker
        self.__ticker_obj = yf.Ticker(ticker)

    def fetch_call_options(self, exp_date):
        return self.__ticker_obj.option_chain(date=exp_date).calls

    def fetch_put_options(self, exp_date):
        return self.__ticker_obj.option_chain(date=exp_date).puts

    def fetch_all_options(self, exp_date):
        return self.__ticker_obj.option_chain(date=exp_date)

    def get_expirations(self):
        return  self.__ticker_obj.options

    def get_earliest_expiration(self):
        return self.__ticker_obj.options[0]

    def get_strike_price(self, exp_date):
        strike_prices = {}
        calls = self.fetch_call_options(exp_date)
        puts = self.fetch_put_options(exp_date)
        strikes = {
            'call' : calls['strike'].tolist(),
            'put' : puts['strike'].tolist(),
        }
        strike_prices[exp_date] = strikes

        return strike_prices

    # maybe add a filter to for in-the-money, at-the-money, out-of-the-money
    def get_nearest_strike_price(self, option_type: OptionType, stock_price, exp_date):

        if option_type == OptionType.EUROPEAN_CALL:
            options = self.fetch_call_options(exp_date)
        elif option_type == OptionType.EUROPEAN_PUT:
            options = self.fetch_put_options(exp_date)
        else:
            raise ValueError(f'Unsupported option type: {option_type}')

        diff = np.abs(options['strike'] - stock_price)
        nearest_idx = np.argmin(diff)
        strike_price = options['strike'].iloc[nearest_idx]
        return strike_price


    def get_all_strike_prices(self):
        strike_prices = {}

        for exp_date in self.get_expirations():
            calls = self.fetch_call_options(exp_date)
            puts = self.fetch_put_options(exp_date)
            strikes = {
                'call' : calls['strike'].tolist(),
                'put' : puts['strike'].tolist(),
            }
            strike_prices[exp_date] = strikes

        return strike_prices




