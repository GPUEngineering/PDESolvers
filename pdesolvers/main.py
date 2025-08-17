import numpy as np
import pandas as pd
import pdesolvers as pde
import cupy as cp

def main():

    # testing for heat equation

    # equation1 = (pde.HeatEquation(1, 100,30,10000, 0.01)
    #             .set_initial_temp(lambda x: np.sin(np.pi * x) + 5)
    #             .set_left_boundary_temp(lambda t: 20 * np.sin(np.pi * t) + 5)
    #             .set_right_boundary_temp(lambda t: t + 5))
    #
    #
    # solver1 = pde.Heat1DCNSolver(equation1)
    # solver2 = pde.Heat1DExplicitSolver(equation1)

    # testing 2d heat equation
    
    xLength = 10  # Lx
    yLength = 10  # Ly
    maxTime = 3  # tmax
    diffusivityConstant = 15  # kappa
    numPointsSpace = 1000  # x_points = y_points
    numPointsTime = 10  # t_points
    ts = 50

    equation = (pde.HeatEquation2D(maxTime, numPointsTime, diffusivityConstant, xLength, numPointsSpace))
    equation.set_initial_temp(lambda x, y: 10 * np.exp(-((x - xLength/2)**2 + (y - yLength/2)**2) / 2))
    equation.set_right_boundary_temp(lambda t, y: min(50, 20 + 5 * y * (yLength - y)**3 * (t/ts)**2))
    equation.set_left_boundary_temp(lambda t, y: min(50, 20 + 10 * y * (yLength - y) * (t/ts)**2))
    equation.set_top_boundary_temp(lambda t, x: min(50,20 + 5 * x * (xLength - x) * (t/ts)**2))
    equation.set_bottom_boundary_temp(lambda t, x: 20)

    # solver1 = pde.Heat2DExplicitSolver(equation)
    # solution1 = solver1.solve()
    # solution1.animate(filename="Explicit")

    # solver2 = pde.Heat2DCNSolver(equation)
    # solution2 = solver2.solve()
    # solution2.animate(#export=True, 
    #                   filename="Crank-Nicolson-400_spatial_pts-CPU")
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
    solver3 = pde.Heat2DCNSolverGPU(equation)
    solution3 = solver3.solve()
    solution3.animate(#export=True,
                      filename="Crank-Nicolson-200_spatial_pts-gpu")
    
    # testing for monte carlo pricing
    # ticker = 'AAPL'

    # # STOCK
    # historical_data = pde.HistoricalStockData(ticker)
    # historical_data.fetch_stock_data( "2024-03-21","2025-03-21")
    # sigma, r = historical_data.estimate_metrics()
    # current_price = historical_data.get_latest_stock_price()

    # equation2 = pde.BlackScholesEquation(pde.OptionType.EUROPEAN_CALL, current_price, 100, r, sigma, 1, 100, 20000)

    # solver1 = pde.BlackScholesCNSolver(equation2)
    # solver2 = pde.BlackScholesExplicitSolver(equation2)
    # sol1 = solver1.solve()
    # sol1.plot()

    # # COMPARISON
    # #  look to see the corresponding option price for the expiration date and strike price
    # pricing_1 = pde.BlackScholesFormula(pde.OptionType.EUROPEAN_CALL, current_price, 100, r, sigma, 1)
    # pricing_2 = pde.MonteCarloPricing(pde.OptionType.EUROPEAN_CALL, current_price, 100, r, sigma, 1, 365, 1000)

    # bs_price = pricing_1.get_black_scholes_merton_price()
    # monte_carlo_price = pricing_2.get_monte_carlo_option_price()

    # pde_price = sol1.get_result()[-1, 0]
    # print(f"PDE Price: {pde_price}")
    # print(f"Black-Scholes Price: {bs_price}")
    # print(f"Monte-Carlo Price: {monte_carlo_price}")

if __name__ == "__main__":
    main()