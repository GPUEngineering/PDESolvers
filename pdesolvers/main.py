import numpy as np
import pandas as pd
import pdesolvers as pde
import matplotlib.pyplot as plt

def plot_frame(solution, frame_index, export=False, filename=None, figsize=(10, 8)):
    """
    Plot a specific frame from the 2D heat equation solution as a 3D surface
    
    :param solution: Heat2DSolution object
    :param frame_index: Time step index to plot (0 to len(t_grid)-1)
    :param export: Whether to save the plot
    :param filename: Filename for export (without extension)
    :param figsize: Figure size tuple
    """
    if frame_index < 0 or frame_index >= len(solution.t_grid):
        raise ValueError(f"Frame index {frame_index} out of range [0, {len(solution.t_grid)-1}]")
    
    print(f"Plotting frame {frame_index} at t = {solution.t_grid[frame_index]:.4f}")
    
    # Set up the figure
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 10
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the temperature data for this frame
    u_k = solution.result[frame_index]
    
    # Create meshgrid
    X, Y = np.meshgrid(solution.x_grid, solution.y_grid)
    
    # Plot surface (transpose to match meshgrid orientation)
    surf = ax.plot_surface(X, Y, u_k.T, 
                          cmap='hot',
                          alpha=0.9,
                          linewidth=0.5,
                          antialiased=True)
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_zlabel('Temperature')
    ax.set_title(f'2D Heat Equation: t = {frame_index * solution.dt:.4f}')
    
    # Dynamic z-limits - rounds up to nearest 50
    ax.set_zlim(0, int(np.ceil(np.max(u_k) / 50) * 50))
    
    # Add colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Temperature')
    
    # Export if requested
    if export:
        if filename is None:
            filename = f"heat_equation_frame_{frame_index:04d}"
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Frame saved as {filename}.png")
    
    # plt.show()
    
    return fig, ax

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
    maxTime = 2  # tmax
    diffusivityConstant = 15  # kappa
    numPointsSpace = 50  # x_points = y_points
    numPointsTime = 2000  # t_points
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
    
    solver2 = pde.Heat2DCNSolver(equation)
    solution2 = solver2.solve()
    # solution2.animate(filename="Crank-Nicolson")
    plot_frame(solution2, 10, export=True, filename="implicit-1_initial_condition")     # t=0
    plot_frame(solution2, 60, export=True, filename="implicit-1_mid_evolution")        # Middle
    plot_frame(solution2, 119, export=True, filename="implicit-1_final_state")        # Final

    
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