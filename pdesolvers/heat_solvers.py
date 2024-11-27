import pdesolvers


class HeatEquationExplicitSolver:

    def __init__(self, grid: pdesolvers.FDMGrid, kappa=1):
        self.__kappa = kappa
        self.__grid = grid

    def solve(self) -> pdesolvers.FDMGrid:
        # This method solves the heat equation. Then, the user can take the grid (using .grid()) and plot it
        pass


class HeatEquationCrankNicolsonSolver:

    def __init__(self, grid: pdesolvers.FDMGrid, kappa=1):
        self.__kappa = kappa
        self.__grid = grid

    def solve(self) -> pdesolvers.FDMGrid:
        pass
