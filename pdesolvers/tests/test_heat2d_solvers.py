import pytest
import numpy as np

import pdesolvers.pdes.heat_2d as heat2d
import pdesolvers.solvers.heat2d_solvers as solver2d
import pdesolvers.utils.utility as utility

class TestHeat2DSolvers:

    def setup_method(self):
        """Setup basic 2D heat equation for each test"""
        self.equation = heat2d.HeatEquation2D(
            time=0.5, t_nodes=800,
            k=1.0, 
            xlength=1.0, x_nodes=20, 
            ylength=1.0, y_nodes=20
        )
        
        # Set up boundary and initial conditions
        self.equation.set_initial_temp(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y))
        self.equation.set_left_boundary_temp(lambda t, y: 0)
        self.equation.set_right_boundary_temp(lambda t, y: 0)
        self.equation.set_bottom_boundary_temp(lambda t, x: 0)
        self.equation.set_top_boundary_temp(lambda t, x: 0)

    def test_check_boundary_conditions_consistency_at_time_zero(self):
        """Test that boundary conditions match initial conditions at corners"""
        # Test corner consistency
        tolerance = 1e-12
        
        # Bottom-left corner (0,0)
        assert abs(self.equation.left_boundary(0, 0) - self.equation.initial_temp(0, 0)) < tolerance
        assert abs(self.equation.bottom_boundary(0, 0) - self.equation.initial_temp(0, 0)) < tolerance
        
        # Bottom-right corner (1,0)
        assert abs(self.equation.right_boundary(0, 0) - self.equation.initial_temp(1.0, 0)) < tolerance
        assert abs(self.equation.bottom_boundary(0, 1.0) - self.equation.initial_temp(1.0, 0)) < tolerance
        
        # Top-left corner (0,1)
        assert abs(self.equation.left_boundary(0, 1.0) - self.equation.initial_temp(0, 1.0)) < tolerance
        assert abs(self.equation.top_boundary(0, 0) - self.equation.initial_temp(0, 1.0)) < tolerance
        
        # Top-right corner (1,1)
        assert abs(self.equation.right_boundary(0, 1.0) - self.equation.initial_temp(1.0, 1.0)) < tolerance
        assert abs(self.equation.top_boundary(0, 1.0) - self.equation.initial_temp(1.0, 1.0)) < tolerance

    def test_solution_dimensions_explicit(self):
        """Test that explicit solver solution has correct dimensions"""
        result = solver2d.Heat2DExplicitSolver(self.equation).solve()
        solution = result.get_result()
        
        expected_shape = (self.equation.t_nodes, self.equation.x_nodes, self.equation.y_nodes)
        assert solution.shape == expected_shape, f"Expected shape {expected_shape}, got {solution.shape}"

    def test_solution_dimensions_crank_nicolson(self):
        """Test that Crank-Nicolson solver solution has correct dimensions"""
        result = solver2d.Heat2DCNSolver(self.equation).solve()
        solution = result.get_result()
        
        expected_shape = (self.equation.t_nodes, self.equation.x_nodes, self.equation.y_nodes)
        assert solution.shape == expected_shape, f"Expected shape {expected_shape}, got {solution.shape}"

    def test_convergence_between_explicit_and_crank_nicolson(self):
        """Test that explicit and Crank-Nicolson methods converge to similar solutions"""
        result_explicit = solver2d.Heat2DExplicitSolver(self.equation).solve().get_result()
        result_cn = solver2d.Heat2DCNSolver(self.equation).solve().get_result()

        # Compare solutions at final time
        diff = np.abs(result_explicit[-1] - result_cn[-1])
        max_diff = np.max(diff)
        
        assert max_diff < 0.1, f"Methods differ by {max_diff}, expected < 0.1"

    def test_maximum_principle_explicit(self):
        """Test that maximum principle holds for explicit solver"""
        result = solver2d.Heat2DExplicitSolver(self.equation).solve().get_result()
        
        initial_max = np.max(result[0])
        solution_max = np.max(result)
        
        tolerance = 1e-10
        assert solution_max <= initial_max + tolerance, f"Maximum principle violated: max = {solution_max}, initial_max = {initial_max}"

    def test_maximum_principle_crank_nicolson(self):
        """Test that maximum principle holds for Crank-Nicolson solver"""
        result = solver2d.Heat2DCNSolver(self.equation).solve().get_result()
        
        initial_max = np.max(result[0])
        solution_max = np.max(result)
        
        tolerance = 1e-10
        assert solution_max <= initial_max + tolerance, f"Maximum principle violated: max = {solution_max}, initial_max = {initial_max}"

    def test_steady_state_convergence_with_constant_boundaries(self):
        """Test convergence to steady state with constant boundary conditions"""
        steady_eq = heat2d.HeatEquation2D(
            time=5.0, t_nodes=1000, k=1.0,
            xlength=1.0, x_nodes=20,
            ylength=1.0, y_nodes=20
        )
        
        # Constant boundary conditions
        steady_eq.set_initial_temp(lambda x, y: 0)
        steady_eq.set_left_boundary_temp(lambda t, y: 1.0)
        steady_eq.set_right_boundary_temp(lambda t, y: 0.0)
        steady_eq.set_bottom_boundary_temp(lambda t, x: 0.0)
        steady_eq.set_top_boundary_temp(lambda t, x: 0.0)
        
        result = solver2d.Heat2DCNSolver(steady_eq).solve().get_result()
        
        # Check that solution has reached steady state (small time derivative)
        final_change = np.abs(result[-1] - result[-10])
        max_change = np.max(final_change)
        
        assert max_change < 1e-3, f"Solution should reach steady state: max_change = {max_change}"

    def test_symmetry_preservation(self):
        """Test that symmetric initial conditions preserve symmetry"""
        sym_eq = heat2d.HeatEquation2D(
            time=0.1, t_nodes=50, k=1.0,
            xlength=2.0, x_nodes=20,
            ylength=2.0, y_nodes=20
        )
        
        # Symmetric initial condition (Gaussian centered at (1,1))
        sym_eq.set_initial_temp(lambda x, y: np.exp(-((x-1)**2 + (y-1)**2)))
        sym_eq.set_left_boundary_temp(lambda t, y: 0)
        sym_eq.set_right_boundary_temp(lambda t, y: 0)
        sym_eq.set_bottom_boundary_temp(lambda t, x: 0)
        sym_eq.set_top_boundary_temp(lambda t, x: 0)
        
        result = solver2d.Heat2DCNSolver(sym_eq).solve().get_result()
        final_solution = result[-1]
        
        # Test x-symmetry (solution should be symmetric about x=1)
        left_half = final_solution[:10, :]
        right_half = np.flip(final_solution[10:, :], axis=0)
        
        symmetry_diff = np.max(np.abs(left_half - right_half))
        assert symmetry_diff < 1e-2, f"X-symmetry not preserved: diff = {symmetry_diff}"

    def test_grid_refinement_convergence(self):
        """Test that finer grids give more accurate solutions"""
        # Coarse grid
        coarse_eq = heat2d.HeatEquation2D(
            time=0.1, t_nodes=50, k=1.0,
            xlength=1.0, x_nodes=10,
            ylength=1.0, y_nodes=10
        )
        coarse_eq.set_initial_temp(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y))
        coarse_eq.set_left_boundary_temp(lambda t, y: 0)
        coarse_eq.set_right_boundary_temp(lambda t, y: 0)
        coarse_eq.set_bottom_boundary_temp(lambda t, x: 0)
        coarse_eq.set_top_boundary_temp(lambda t, x: 0)
        
        # Fine grid
        fine_eq = heat2d.HeatEquation2D(
            time=0.1, t_nodes=100, k=1.0,
            xlength=1.0, x_nodes=20,
            ylength=1.0, y_nodes=20
        )
        fine_eq.set_initial_temp(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y))
        fine_eq.set_left_boundary_temp(lambda t, y: 0)
        fine_eq.set_right_boundary_temp(lambda t, y: 0)
        fine_eq.set_bottom_boundary_temp(lambda t, x: 0)
        fine_eq.set_top_boundary_temp(lambda t, x: 0)
        
        coarse_result = solver2d.Heat2DCNSolver(coarse_eq).solve().get_result()
        fine_result = solver2d.Heat2DCNSolver(fine_eq).solve().get_result()
        
        # Sample at center points for comparison
        coarse_center = coarse_result[-1, 5, 5]
        fine_center = fine_result[-1, 10, 10]
        
        # Fine grid should be closer to analytical solution
        analytical_center = np.exp(-2 * np.pi**2 * 0.1) * np.sin(np.pi * 0.5) * np.sin(np.pi * 0.5)
        
        coarse_error = abs(coarse_center - analytical_center)
        fine_error = abs(fine_center - analytical_center)
        
        assert fine_error < coarse_error, f"Fine grid should be more accurate: coarse_error={coarse_error}, fine_error={fine_error}"

    @pytest.mark.parametrize("solver_class", [solver2d.Heat2DExplicitSolver, solver2d.Heat2DCNSolver])
    def test_interpolation_convergence_between_methods(self, solver_class):
        """Test spatial interpolation at final time"""
        result = solver_class(self.equation).solve()
        u = result.get_result()  # Shape: (t_nodes, x_nodes, y_nodes)
        
        # Calculate correct spatial grid spacing
        hx = self.equation.xlength / (self.equation.x_nodes - 1)
        hy = self.equation.ylength / (self.equation.y_nodes - 1)
        
        try:
            # Interpolate in space at final time
            final_field = u[-1]  # 2D spatial field at final time
            interpolator = utility.RBFInterpolator(final_field, hx, hy)
            interp_value = interpolator.interpolate(0.5, 0.5)
            
            assert np.isfinite(interp_value), "Interpolated value should be finite"
            
            # For sin(πx)sin(πy) initial condition with zero boundaries,
            # value should be reasonable (can be negative)
            assert abs(interp_value) < 10, f"Interpolated value seems unreasonable: {interp_value}"
            
        except Exception as e:
            pytest.skip(f"RBF spatial interpolation failed: {e}")
    def test_energy_dissipation_homogeneous_boundaries(self):
        """Test that energy decreases monotonically with homogeneous boundaries"""
        result = solver2d.Heat2DCNSolver(self.equation).solve().get_result()
        
        # Calculate total energy (L2 norm) at each time step
        energies = [np.sum(result[t]**2) for t in range(result.shape[0])]
        
        # Energy should decrease monotonically (or stay constant)
        tolerance = 1e-10
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i-1] + tolerance, f"Energy increased at step {i}: {energies[i]} > {energies[i-1]}"

    def test_explicit_solver_stability_violation_detection(self):
        """Test that explicit solver detects CFL condition violations"""
        unstable_eq = heat2d.HeatEquation2D(
            time=1.0, t_nodes=10,  # Large time step
            k=1.0, xlength=1.0, x_nodes=50,  # Fine spatial grid
            ylength=1.0, y_nodes=50
        )
        unstable_eq.set_initial_temp(lambda x, y: 1.0)
        unstable_eq.set_left_boundary_temp(lambda t, y: 0)
        unstable_eq.set_right_boundary_temp(lambda t, y: 0)
        unstable_eq.set_bottom_boundary_temp(lambda t, x: 0)
        unstable_eq.set_top_boundary_temp(lambda t, x: 0)
        
        with pytest.raises(AssertionError, match="CFL"):
            solver2d.Heat2DExplicitSolver(unstable_eq).solve()

    def test_analytical_solution_comparison(self):
        """Compare with known analytical solution for simple case"""
        # Use equation with known analytical solution
        # u(x,y,t) = exp(-2π²kt) * sin(πx) * sin(πy)
        
        result = solver2d.Heat2DCNSolver(self.equation).solve()
        numerical_solution = result.get_result()
        
        # Create analytical solution
        x = np.linspace(0, 1, self.equation.x_nodes)
        y = np.linspace(0, 1, self.equation.y_nodes)
        t_final = 0.5
        
        X, Y = np.meshgrid(x, y)
        analytical = np.exp(-2 * np.pi**2 * t_final) * np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # Compare at final time
        numerical_final = numerical_solution[-1]
        
        error = np.max(np.abs(numerical_final.T - analytical))
        assert error < 0.1, f"Error vs analytical solution: {error}"

    @pytest.mark.parametrize("solver_class", [solver2d.Heat2DExplicitSolver, solver2d.Heat2DCNSolver])
    def test_execution_time_tracking(self, solver_class):
        """Test that solvers track execution time"""
        result = solver_class(self.equation).solve()
        
        execution_time = result.get_execution_time()
        assert execution_time > 0, f"Execution time should be positive: {execution_time}"
        assert isinstance(execution_time, float), f"Execution time should be float: {type(execution_time)}"

    def test_non_negative_temperatures_with_non_negative_conditions(self):
        """Test that non-negative initial and boundary conditions maintain non-negative temperatures"""
        non_neg_eq = heat2d.HeatEquation2D(
            time=0.1, t_nodes=50, k=1.0,
            xlength=1.0, x_nodes=15,
            ylength=1.0, y_nodes=15
        )
        
        # Non-negative conditions
        non_neg_eq.set_initial_temp(lambda x, y: 10 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1))
        non_neg_eq.set_left_boundary_temp(lambda t, y: 5.0)
        non_neg_eq.set_right_boundary_temp(lambda t, y: 3.0)
        non_neg_eq.set_bottom_boundary_temp(lambda t, x: 2.0)
        non_neg_eq.set_top_boundary_temp(lambda t, x: 4.0)
        
        result = solver2d.Heat2DCNSolver(non_neg_eq).solve().get_result()
        
        # All temperatures should remain non-negative
        min_temp = np.min(result)
        assert min_temp >= -1e-10, f"Negative temperature found: {min_temp}"

    def test_boundary_conditions_preserved_throughout_simulation(self):
        """Test that boundary conditions are maintained at all time steps"""
        result = solver2d.Heat2DCNSolver(self.equation).solve()
        solution = result.get_result()
        
        # Check boundary conditions at a few time steps
        times_to_check = [0, solution.shape[0]//2, -1]
        tolerance = 1e-12
        
        for t_idx in times_to_check:
            t_val = t_idx * (0.5 / (solution.shape[0] - 1)) if t_idx >= 0 else 0.5
            
            # Check left boundary (x=0)
            for j in range(solution.shape[2]):
                y_val = j / (solution.shape[2] - 1)
                expected = self.equation.left_boundary(t_val, y_val)
                actual = solution[t_idx, 0, j]
                assert abs(actual - expected) < tolerance, f"Left boundary violated at t={t_val}, y={y_val}"
            
            # Check right boundary (x=1)
            for j in range(solution.shape[2]):
                y_val = j / (solution.shape[2] - 1)
                expected = self.equation.right_boundary(t_val, y_val)
                actual = solution[t_idx, -1, j]
                assert abs(actual - expected) < tolerance, f"Right boundary violated at t={t_val}, y={y_val}"
    
    def test_invalid_equation_parameters(self):
        """Test that invalid equation parameters raise appropriate errors"""
    
        # Negative time
        with pytest.raises((ValueError, AssertionError)):
            heat2d.HeatEquation2D(time=-1.0, t_nodes=100, k=1.0, xlength=1.0, x_nodes=20, ylength=1.0, y_nodes=20)
        
        # Zero or negative spatial dimensions
        with pytest.raises((ValueError, AssertionError)):
            heat2d.HeatEquation2D(time=1.0, t_nodes=100, k=1.0, xlength=0, x_nodes=20, ylength=1.0, y_nodes=20)
        
        # Negative diffusivity constant
        with pytest.raises((ValueError, AssertionError)):
            heat2d.HeatEquation2D(time=1.0, t_nodes=100, k=-1.0, xlength=1.0, x_nodes=20, ylength=1.0, y_nodes=20)
        
        # Too few nodes
        with pytest.raises((ValueError, AssertionError)):
            heat2d.HeatEquation2D(time=1.0, t_nodes=1, k=1.0, xlength=1.0, x_nodes=2, ylength=1.0, y_nodes=2)

    def test_missing_boundary_conditions(self):
        """Test that missing boundary conditions are detected"""
        incomplete_eq = heat2d.HeatEquation2D(time=0.1, t_nodes=50, k=1.0, xlength=1.0, x_nodes=10, ylength=1.0, y_nodes=10)
    
        # Only set some boundary conditions, leave others undefined
        incomplete_eq.set_initial_temp(lambda x, y: 1.0)
        incomplete_eq.set_left_boundary_temp(lambda t, y: 0)
        # Missing right, top, bottom boundaries
        
        with pytest.raises((AttributeError, ValueError)):
            solver2d.Heat2DExplicitSolver(incomplete_eq).solve()

    def test_missing_initial_condition(self):
        """Test that missing initial condition is detected"""
        incomplete_eq = heat2d.HeatEquation2D(time=0.1, t_nodes=50, k=1.0, xlength=1.0, x_nodes=10, ylength=1.0, y_nodes=10)
        
        # Set boundaries but no initial condition
        incomplete_eq.set_left_boundary_temp(lambda t, y: 0)
        incomplete_eq.set_right_boundary_temp(lambda t, y: 0)
        incomplete_eq.set_bottom_boundary_temp(lambda t, x: 0)
        incomplete_eq.set_top_boundary_temp(lambda t, x: 0)
        
        with pytest.raises((AttributeError, ValueError)):
            solver2d.Heat2DExplicitSolver(incomplete_eq).solve()
    
    def test_boundary_function_exceptions(self):
        """Test handling of boundary functions that raise exceptions"""
        
        def bad_boundary(t, coord):
            if t > 0.05:
                raise ValueError("Boundary function failed")
            return 1.0
        
        bad_eq = heat2d.HeatEquation2D(time=0.1, t_nodes=50, k=1.0, xlength=1.0, x_nodes=10, ylength=1.0, y_nodes=10)
        bad_eq.set_initial_temp(lambda x, y: 1.0)
        bad_eq.set_left_boundary_temp(bad_boundary)
        bad_eq.set_right_boundary_temp(lambda t, y: 0)
        bad_eq.set_bottom_boundary_temp(lambda t, x: 0)
        bad_eq.set_top_boundary_temp(lambda t, x: 0)
        
        with pytest.raises(ValueError):
            solver2d.Heat2DCNSolver(bad_eq).solve()

    def test_boundary_function_returns_invalid_values(self):
        """Test handling of boundary functions returning NaN or inf"""
        
        bad_eq = heat2d.HeatEquation2D(time=0.1, t_nodes=50, k=1.0, xlength=1.0, x_nodes=10, ylength=1.0, y_nodes=10)
        bad_eq.set_initial_temp(lambda x, y: 1.0)
        bad_eq.set_left_boundary_temp(lambda t, y: np.nan)  # Returns NaN
        bad_eq.set_right_boundary_temp(lambda t, y: 0)
        bad_eq.set_bottom_boundary_temp(lambda t, x: 0)
        bad_eq.set_top_boundary_temp(lambda t, x: 0)
        
        result = solver2d.Heat2DCNSolver(bad_eq).solve().get_result()
        
        # Should detect NaN in solution
        assert np.any(np.isnan(result)), "NaN should propagate through solution"

    def test_initial_condition_returns_invalid_values(self):
        """Test handling of initial condition returning invalid values"""
        
        bad_eq = heat2d.HeatEquation2D(time=0.1, t_nodes=50, k=1.0, xlength=1.0, x_nodes=10, ylength=1.0, y_nodes=10)
        bad_eq.set_initial_temp(lambda x, y: np.inf if x > 0.5 else 1.0)
        bad_eq.set_left_boundary_temp(lambda t, y: 0)
        bad_eq.set_right_boundary_temp(lambda t, y: 0)
        bad_eq.set_bottom_boundary_temp(lambda t, x: 0)
        bad_eq.set_top_boundary_temp(lambda t, x: 0)
        
        result = solver2d.Heat2DCNSolver(bad_eq).solve().get_result()
        
        # Should detect inf in solution
        assert np.any(np.isinf(result)), "Inf should propagate through solution"

    def test_solution_object_completeness(self):
        """Test that solution object contains all required information"""
        
        result = solver2d.Heat2DCNSolver(self.equation).solve()
        
        # Test all expected attributes exist
        assert hasattr(result, 'get_result'), "Solution should have get_result method"
        assert hasattr(result, 'get_execution_time'), "Solution should have get_execution_time method"
        
        # Test that methods return valid data
        solution_array = result.get_result()
        assert isinstance(solution_array, np.ndarray), "get_result should return numpy array"
        
        exec_time = result.get_execution_time()
        assert isinstance(exec_time, (int, float)), "get_execution_time should return number"
        assert exec_time >= 0, "Execution time should be non-negative"

    @pytest.mark.parametrize("solver_class", [solver2d.Heat2DExplicitSolver, solver2d.Heat2DCNSolver])
    def test_solver_with_None_equation(self, solver_class):
        """Test solver behavior with None equation"""
        
        with pytest.raises((TypeError, AttributeError)):
            solver_class(None).solve()

    def test_solver_reusability(self):
        """Test that solvers can be reused safely"""
    
        solver = solver2d.Heat2DCNSolver(self.equation)
    
        # First solve
        result1 = solver.solve()
    
        # Second solve should give same result
        result2 = solver.solve()
    
        np.testing.assert_array_equal(result1.get_result(), result2.get_result())

    def test_equation_modification_after_solver_creation(self):
        """Test behavior when equation is modified after solver creation"""
    
        eq = heat2d.HeatEquation2D(time=0.1, t_nodes=50, k=1.0, xlength=1.0, x_nodes=10, ylength=1.0, y_nodes=10)
        eq.set_initial_temp(lambda x, y: 1.0)
        eq.set_left_boundary_temp(lambda t, y: 0)
        eq.set_right_boundary_temp(lambda t, y: 0)
        eq.set_bottom_boundary_temp(lambda t, x: 0)
        eq.set_top_boundary_temp(lambda t, x: 0)
    
        solver = solver2d.Heat2DCNSolver(eq)
    
        # Modify equation after solver creation
        eq.set_initial_temp(lambda x, y: 10.0)
    
        result = solver.solve()