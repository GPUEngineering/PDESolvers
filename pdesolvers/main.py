import numpy as np
import pdesolvers as pde


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

    # testing for bse
    equation2 = pde.BlackScholesEquation('call', 300, 1, 0.2, 0.05, 100, 100, 20000)

    solver1 = pde.BlackScholesCNSolver(equation2)
    solver2 = pde.BlackScholesExplicitSolver(equation2)
    res1 = solver1.solve().get_result()
    res2 = solver2.solve().get_result()

    interpolator1 = pde.RBFInterpolator(res1, 0.8, 200,0.1, 0.03)
    interpolator2 = pde.RBFInterpolator(res2, 0.8, 200,0.1, 0.03)
    print(interpolator1.rbf_interpolate())
    print(interpolator2.rbf_interpolate())


# print(res.shape)
#     res1.plot()

if __name__ == "__main__":
    main()