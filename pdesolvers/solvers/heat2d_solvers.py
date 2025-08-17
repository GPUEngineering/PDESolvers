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
        
        if (self.equation.left_boundary is None or 
            self.equation.right_boundary is None or 
            self.equation.top_boundary is None or 
            self.equation.bottom_boundary is None):
            raise ValueError("All boundary conditions must be set before solving")
        if self.equation.initial_temp is None:
            raise ValueError("Initial condition must be set before solving")

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
        
        print(f"Calculating temperature evolution with {self.equation.t_nodes-1} iterations...", flush=True)
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

        if (self.equation.left_boundary is None or 
            self.equation.right_boundary is None or 
            self.equation.top_boundary is None or 
            self.equation.bottom_boundary is None):
            raise ValueError("All boundary conditions must be set before solving")
        if self.equation.initial_temp is None:
            raise ValueError("Initial condition must be set before solving")

        x = np.linspace(0, self.equation.xlength, self.equation.x_nodes)
        y = np.linspace(0, self.equation.ylength, self.equation.y_nodes)
        t = np.linspace(0, self.equation.time, self.equation.t_nodes)

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dt = t[1]-t[0]
        c = self.equation.k * dt / 2
        cx = c / (dx**2)
        cy = c / (dy**2)
        alpha = 1 + 2*cx + 2*cy
        beta = 1 - 2*cx - 2*cy

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
        G, n_interior_x, n_interior_y = utility.Heat2DHelper.innitTriDiagMatrix(self.equation.x_nodes, self.equation.y_nodes, cx, cy, alpha)
        
        # time-stepping loop
        print(f"Calculating temperature evolution with {self.equation.t_nodes-1} iterations...", flush=True)
        for tau in range(self.equation.t_nodes - 1):
            rhs = np.zeros(n_interior_x * n_interior_y)
            idx = 0
            for j in range(1, self.equation.y_nodes-1):
                for i in range(1, self.equation.x_nodes-1):
                    # RHS = β*U_τ + cx*(neighbors_x) + cy*(neighbors_y) + boundary_terms
                    rhs[idx] = beta * U[tau, i, j]
                    rhs[idx] += cx * (U[tau, i-1, j] + U[tau, i+1, j])
                    rhs[idx] += cy * (U[tau, i, j-1] + U[tau, i, j+1])
                    # Boundary contributions
                    if i == 1:
                        rhs[idx] += cx * U[tau+1, 0, j]
                    if i == self.equation.x_nodes-2:
                        rhs[idx] += cx * U[tau+1, -1, j]
                    if j == 1:
                        rhs[idx] += cy * U[tau+1, i, 0]
                    if j == self.equation.y_nodes-2:
                        rhs[idx] += cy * U[tau+1, i, -1]
                    idx += 1
            # Solve G*u_{τ+1} = rhs
            u_next_interior = spsolve(G, rhs)
            U[tau+1, 1:-1, 1:-1] = u_next_interior.reshape((n_interior_x, n_interior_y))
            
        end = time.perf_counter()
        duration = end - start
        logging.info(f"Solver completed in {duration} seconds.")
        return sol.Heat2DSolution(U, x, y, t, dx, dy, dt, duration)