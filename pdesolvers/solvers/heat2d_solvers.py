import time
import logging

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
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
    
class Heat2DCNSolverGPU(Solver):
    """
    2D Heat Equation Crank-Nicolson solver with CuPy GPU acceleration
    Maintains the same interface as existing solvers but uses GPU computation
    """
    
    def __init__(self, equation: heat.HeatEquation2D):
        self.equation = equation

    def solve(self):
        """
        Solves the 2D heat equation using GPU-accelerated Crank-Nicolson method
        
        Returns:
            Heat2DSolution: Solution object with results and metadata
        """
        logging.info(f"Starting {self.__class__.__name__} with {self.equation.x_nodes} x {self.equation.y_nodes} spatial nodes and {self.equation.t_nodes} time nodes.")
        start = time.perf_counter()

        # Validate boundary conditions
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

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dt = t[1] - t[0]
        c = self.equation.k * dt / 2
        cx = c / (dx**2)
        cy = c / (dy**2)
        alpha = 1 + 2*cx + 2*cy
        beta = 1 - 2*cx - 2*cy

        logging.info(f"GPU acceleration enabled. Stability parameters: cx={cx:.6f}, cy={cy:.6f}")

        # Pre-compute ALL boundary values on CPU, then transfer to GPU once
        print("Pre-computing boundary conditions...")
        boundary_data = utility.Heat2DHelper.precompute_boundaries_cpu(self, x, y, t)
        
        # Transfer everything to GPU in one go
        print("Transferring data to GPU...")
        gpu_data = utility.Heat2DHelper.transfer_to_gpu(self, boundary_data, x, y, t)
        
        # Build sparse system matrix on GPU
        print("Building sparse matrix on GPU...")
        G = utility.Heat2DHelper.build_gpu_sparse_matrix(self, cx, cy, alpha)
        
        # Initialize solution on GPU
        U_gpu = utility.Heat2DHelper.initialize_solution_gpu(self, gpu_data)

        # GPU time-stepping loop
        print(f"Calculating temperature evolution on GPU with {self.equation.t_nodes-1} iterations...", flush=True)
        nx, ny, nt = self.equation.x_nodes, self.equation.y_nodes, self.equation.t_nodes
        n_interior_x = nx - 2
        n_interior_y = ny - 2
        
        # Pre-allocate RHS vector on GPU
        rhs = cp.zeros(n_interior_x * n_interior_y)
        
        for tau in range(nt - 1):
            # Vectorized RHS computation using GPU array operations
            interior_current = U_gpu[tau, 1:-1, 1:-1]  # Current interior points
            
            # Vectorized finite difference stencil
            rhs_2d = (beta * interior_current + 
                     cx * (U_gpu[tau, :-2, 1:-1] + U_gpu[tau, 2:, 1:-1]) +  # x-neighbors
                     cy * (U_gpu[tau, 1:-1, :-2] + U_gpu[tau, 1:-1, 2:]))   # y-neighbors
            
            # Add boundary contributions (vectorized)
            # Left boundary contribution
            rhs_2d[0, :] += cx * U_gpu[tau+1, 0, 1:-1]
            # Right boundary contribution  
            rhs_2d[-1, :] += cx * U_gpu[tau+1, -1, 1:-1]
            # Bottom boundary contribution
            rhs_2d[:, 0] += cy * U_gpu[tau+1, 1:-1, 0]
            # Top boundary contribution
            rhs_2d[:, -1] += cy * U_gpu[tau+1, 1:-1, -1]
            
            # Flatten for sparse solve
            rhs = rhs_2d.flatten()
            
            # Solve linear system on GPU
            u_next_interior = cupy_spsolve(G, rhs)
            
            # Reshape and assign
            U_gpu[tau+1, 1:-1, 1:-1] = u_next_interior.reshape((n_interior_x, n_interior_y))

        # Transfer final result back to CPU
        U_cpu = cp.asnumpy(U_gpu)
        x_cpu = cp.asnumpy(x) if isinstance(x, cp.ndarray) else x
        y_cpu = cp.asnumpy(y) if isinstance(y, cp.ndarray) else y
        t_cpu = cp.asnumpy(t) if isinstance(t, cp.ndarray) else t

        end = time.perf_counter()
        duration = end - start
        logging.info(f"GPU solver completed in {duration} seconds.")
        
        return sol.Heat2DSolution(U_cpu, x_cpu, y_cpu, t_cpu, dx, dy, dt, duration)