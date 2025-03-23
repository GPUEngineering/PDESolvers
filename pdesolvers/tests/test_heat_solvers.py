import pytest
import numpy as np

import pdesolvers.pdes.heat_1d as heat
import pdesolvers.solvers.heat_solvers as solver
import pdesolvers.utils.utility as utility

class TestHeatSolvers:

    def setup_method(self):
        self.equation = (heat.HeatEquation(1, 100,30,10000, 0.01)
                 .set_initial_temp(lambda x: 10 * np.sin(2 * np.pi * x) + 15)
                 .set_left_boundary_temp(lambda t: t + 15)
                 .set_right_boundary_temp(lambda t: 15))

    def test_check_terminal_and_boundary_conditions_at_time_zero(self):
        length = self.equation.length
        assert abs(self.equation.get_left_boundary(0) - self.equation.get_initial_temp(0)) < 1e-12, "Left boundary condition failed"
        assert abs(self.equation.get_right_boundary(0) - self.equation.get_initial_temp(length)) < 1e-12, "Right boundary condition failed"

    # explicit method tests

    def test_check_absolute_difference_between_two_results(self):
        result1 = solver.Heat1DExplicitSolver(self.equation).solve().get_result()
        result2 = solver.Heat1DCNSolver(self.equation).solve().get_result()

        diff = result1 - result2

        assert np.max(np.abs(diff)) < 1e-2

    def test_convergence_between_single_interpolated_point(self):
        result1 = solver.Heat1DExplicitSolver(self.equation).solve()
        result2 = solver.Heat1DCNSolver(self.equation).solve()
        u1 = result1.get_result()
        u2 = result2.get_result()

        data1 = utility.RBFInterpolator(u1, 0.1, 0.03).interpolate(0.2,0.9)
        data2 = utility.RBFInterpolator(u2, 0.1, 0.03).interpolate(0.2,0.9)

        diff = np.abs(data1 - data2)

        assert diff < 1e-4

    def test_convergence_between_two_interpolated_grids(self):
        result1 = solver.Heat1DExplicitSolver(self.equation).solve()
        result2 = solver.Heat1DCNSolver(self.equation).solve()

        diff = np.abs(result1 - result2)

        assert diff < 1e-1

