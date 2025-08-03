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
        
        end = time.perf_counter()
        duration = end - start
        logging.info(f"Solver completed in {duration} seconds.")
        return None
    
class Heat2DCNSolver (Solver):
    def __init__(self, equation: heat.HeatEquation2D):
        self.equation = equation

    def solve(self):
        logging.info(f"Starting {self.__class__.__name__} with {self.equation.x_nodes+1} spatial nodes and {self.equation.t_nodes+1} time nodes.")
        start = time.perf_counter()
        
        end = time.perf_counter()
        duration = end - start
        logging.info(f"Solver completed in {duration} seconds.")
        return None