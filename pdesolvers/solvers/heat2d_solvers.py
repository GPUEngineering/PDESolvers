import time
import logging

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
from cupyx.scipy.sparse import diags as cupy_diags
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
        logging.info(f"Starting {self.__class__.__name__} with {self.equation.x_nodes} spatial nodes and {self.equation.t_nodes} time nodes.")
        start = time.perf_counter()

        # Validation
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
        
        # Crank-Nicolson parameters (c = κ*dt/2)
        c = self.equation.k * dt / 2
        cx = c / (dx**2)
        cy = c / (dy**2)
        alpha = 1 + 2*cx + 2*cy  # Diagonal coefficient for LHS
        beta = 1 - 2*cx - 2*cy   # Diagonal coefficient for RHS

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
        
        # Build the LHS matrix (I - c*Δ̃)
        n_interior_x = self.equation.x_nodes - 2
        n_interior_y = self.equation.y_nodes - 2
        n_total = n_interior_x * n_interior_y
        
        # Create the sparse matrix for the linear system
        G = utility.Heat2DHelper.build_cn_matrix(n_interior_x, n_interior_y, cx, cy, alpha)
        
        # Time-stepping loop
        print(f"Calculating temperature evolution with {self.equation.t_nodes-1} iterations...")
        for tau in range(self.equation.t_nodes - 1):
            # Build RHS vector: (I + c*Δ̃)U_τ + boundary terms
            rhs = np.zeros(n_total)
            idx = 0
            
            for j in range(1, self.equation.y_nodes-1):
                for i in range(1, self.equation.x_nodes-1):
                    # RHS from (I + c*Δ̃)U_τ
                    rhs[idx] = (beta * U[tau, i, j] + 
                            cx * (U[tau, i-1, j] + U[tau, i+1, j]) +
                            cy * (U[tau, i, j-1] + U[tau, i, j+1]))
                    
                    # Add boundary contributions from τ+1
                    if i == 1:  # Left boundary
                        rhs[idx] += cx * U[tau+1, 0, j]
                    if i == self.equation.x_nodes-2:  # Right boundary  
                        rhs[idx] += cx * U[tau+1, -1, j]
                    if j == 1:  # Bottom boundary
                        rhs[idx] += cy * U[tau+1, i, 0]
                    if j == self.equation.y_nodes-2:  # Top boundary
                        rhs[idx] += cy * U[tau+1, i, -1]
                        
                    idx += 1
            
            # Solve the linear system: G * u_{τ+1} = rhs
            u_next_interior = spsolve(G, rhs)
            
            # Reshape and assign to interior points
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
        Mirrors the CPU implementation exactly but on GPU
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
        print(f"CuPy using device: {cp.cuda.Device()}")
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

        print("Initializing matrix on GPU...")
        # Initialize matrix on GPU using same utility function but transfer to GPU
        U_cpu = utility.Heat2DHelper.initMatrix(self.equation.t_nodes,
                                               self.equation.x_nodes,
                                               self.equation.y_nodes,
                                               self.equation.left_boundary,
                                               self.equation.right_boundary,
                                               self.equation.bottom_boundary,
                                               self.equation.top_boundary,
                                               self.equation.initial_temp,
                                               x, y, t)
        
        print("=== GPU Solver Debug Start ===")
        print("About to transfer U_cpu to GPU...")
        U_gpu = cp.asarray(U_cpu)
        print("Matrix transferred to GPU")
        print("About to build sparse matrix on GPU...")
        
        # Build sparse matrix on GPU (same structure as CPU)
        n_interior_x = self.equation.x_nodes - 2
        n_interior_y = self.equation.y_nodes - 2
        n_total = n_interior_x * n_interior_y
        
        # Build matrix directly on GPU
        main_diag = cp.full(n_total, alpha)
        x_off_diag = cp.full(n_total - 1, -cx)
        y_off_diag = cp.full(n_total - n_interior_x, -cy)
        
        # Zero boundary crossings
        for i in range(n_interior_x - 1, n_total - 1, n_interior_x):
            x_off_diag[i] = 0
        
        
        G = cupy_diags([y_off_diag, x_off_diag, main_diag, x_off_diag, y_off_diag], 
                       offsets=[-n_interior_x, -1, 0, 1, n_interior_x], 
                       shape=(n_total, n_total), format='csr')
        
        # Time-stepping loop
        print(f"Calculating temperature evolution on GPU with {self.equation.t_nodes-1} iterations...")
        for tau in range(self.equation.t_nodes - 1):
            print(f'calculating temperature at time-step: {tau}', flush=True)
            # Build RHS vector: (I + c*Δ̃)U_τ + boundary terms
            rhs = cp.zeros(n_total)
            
            # RHS calculation with loops
            # idx = 0
            
            # for j in range(1, self.equation.y_nodes-1):
            #     for i in range(1, self.equation.x_nodes-1):
            #         # RHS from (I + c*Δ̃)U_τ - SAME AS CPU
            #         rhs[idx] = (beta * U_gpu[tau, i, j] + 
            #                   cx * (U_gpu[tau, i-1, j] + U_gpu[tau, i+1, j]) +
            #                   cy * (U_gpu[tau, i, j-1] + U_gpu[tau, i, j+1]))
                    
            #         # Add boundary contributions from τ+1 - SAME AS CPU
            #         if i == 1:  # Left boundary
            #             rhs[idx] += cx * U_gpu[tau+1, 0, j]
            #         if i == self.equation.x_nodes-2:  # Right boundary  
            #             rhs[idx] += cx * U_gpu[tau+1, -1, j]
            #         if j == 1:  # Bottom boundary
            #             rhs[idx] += cy * U_gpu[tau+1, i, 0]
            #         if j == self.equation.y_nodes-2:  # Top boundary
            #             rhs[idx] += cy * U_gpu[tau+1, i, -1]
                        
            #         idx += 1

            # Vectorized RHS computation
            U_interior = U_gpu[tau, 1:-1, 1:-1]
            U_left = U_gpu[tau, :-2, 1:-1] 
            U_right = U_gpu[tau, 2:, 1:-1]
            U_bottom = U_gpu[tau, 1:-1, :-2]
            U_top = U_gpu[tau, 1:-1, 2:]

            rhs_matrix = (beta * U_interior + 
                        cx * (U_left + U_right) + 
                        cy * (U_bottom + U_top))

            # Add boundary terms vectorized
            rhs_matrix[:, 0] += cx * U_gpu[tau+1, 1:-1, 0]   # Left boundary
            rhs_matrix[:, -1] += cx * U_gpu[tau+1, 1:-1, -1] # Right boundary  
            rhs_matrix[0, :] += cy * U_gpu[tau+1, 0, 1:-1]   # Bottom boundary
            rhs_matrix[-1, :] += cy * U_gpu[tau+1, -1, 1:-1] # Top boundary

            rhs = rhs_matrix.flatten()
            
            # Solve the linear system on GPU
            u_next_interior = cupy_spsolve(G, rhs)
            
            # Reshape and assign to interior points
            U_gpu[tau+1, 1:-1, 1:-1] = u_next_interior.reshape((n_interior_x, n_interior_y))

        # Transfer final result back to CPU
        U_cpu = cp.asnumpy(U_gpu)

        end = time.perf_counter()
        duration = end - start
        logging.info(f"GPU solver completed in {duration} seconds.")
        
        return sol.Heat2DSolution(U_cpu, x, y, t, dx, dy, dt, duration)