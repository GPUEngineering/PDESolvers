import time
import logging

import numpy as np
import pdesolvers.solution as sol
import pdesolvers.pdes.heat_2d as heat
import pdesolvers.utils.utility as utility

from scipy.sparse.linalg import spsolve
from pdesolvers.solvers.solver import Solver

logging.basicConfig(
    level = logging.INFO,
    format = "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

class Heat2DExplicitSolver (Solver):
    def __init__(self, equation: heat.HeatEquation2D):
        self.equation = equation

    def solve(self):
        logging.info(f"Starting {self.__class__.__name__} with {self.equation.x_nodes+1} spatial nodes and {self.equation.t_nodes+1} time nodes.")
        start = time.perf_counter()
        
        x = np.linspace(0, self.equation.xlength, self.equation.x_nodes)
        y = np.linspace(0, self.equation.ylength, self.equation.y_nodes)
        t = np.linspace(0, self.equation.time, self.equation.t_nodes)

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dt = t[1]-t[0]
        assert dt < (dx*dy)/(4*self.equation.k), "Time-step size too large!! violates CFL condtion for Forward Euler method."
        lambdaConstant = (self.equation.k * dt)

        print("Initializing matrix...")
        U = utility.Heat2DHelper.initMatrix(self.equation.t_nodes,
                                                 self.equation.x_nodes,
                                                 self.equation.y_nodes,
                                                 self.equation.left_boundary,
                                                 self.equation.right_boundary,
                                                 self.equation.bottom_boundary,
                                                 self.equation.top_boundary,
                                                 self.equation.initial_temp,
                                                 x, y, t)
        
        print("Calculating temperature evolution...")
        for tau in range(self.equation.t_nodes-1):
            for i in range(1, self.equation.x_nodes-1):
                for j in range(1, self.equation.y_nodes-1):
                    # 5-point stencil for 2D heat equation
                    U[tau+1, i, j] = U[tau, i, j] + lambdaConstant * (
                        # Second derivative in x-direction
                        (1/dx**2)*(U[tau, i-1, j] - 2*U[tau, i, j] + U[tau, i+1, j]) +
                        # Second derivative in y-direction  
                        (1/dy**2)*(U[tau, i, j-1] - 2*U[tau, i, j] + U[tau, i, j+1])
                    )
        
        end = time.perf_counter()
        duration = end - start
        logging.info(f"Solver completed in {duration} seconds.")
        return sol.Heat2DSolution(U, x, y, t, dx, dy, dt, duration)
    
class Heat2DCNSolver (Solver):
    def __init__(self, equation: heat.HeatEquation2D):
        self.equation = equation

    def solve(self):
        logging.info(f"Starting {self.__class__.__name__} with {self.equation.x_nodes+1} spatial nodes and {self.equation.t_nodes+1} time nodes.")
        start = time.perf_counter()
        x = np.linspace(0, self.equation.xlength, self.equation.x_nodes)
        y = np.linspace(0, self.equation.ylength, self.equation.y_nodes)
        t = np.linspace(0, self.equation.time, self.equation.t_nodes)

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dt = t[1]-t[0]

        print("Initializing matrix...")
        U = utility.Heat2DHelper.initMatrix(self.equation.t_nodes,
                                                 self.equation.x_nodes,
                                                 self.equation.y_nodes,
                                                 self.equation.left_boundary,
                                                 self.equation.right_boundary,
                                                 self.equation.bottom_boundary,
                                                 self.equation.top_boundary,
                                                 self.equation.initial_temp,
                                                 x, y, t)
        
        # create sparse matrix

        # time-stepping loop
        
        end = time.perf_counter()
        duration = end - start
        logging.info(f"Solver completed in {duration} seconds.")
        return sol.Heat2DSolution(U, x, y, t, dx, dy, dt, duration)