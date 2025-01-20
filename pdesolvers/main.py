import numpy as np
import pdesolvers as pde


def main():

    # testing for heat equation

    equation1 = (pde.HeatEquation(1, 100,30,10000, 0.01)
                .set_initial_temp(lambda x: np.sin(np.pi * x) + 5)
                .set_left_boundary_temp(lambda t: 20 * np.sin(np.pi * t) + 5)
                .set_right_boundary_temp(lambda t: t + 5))


    # solver1 = pde.Heat1DCNSolver(equation1)
    # solver2 = pde.Heat1DExplicitSolver(equation1)

    # testing for bse
    equation2 = pde.BlackScholesEquation('call', 300, 1, 0.2, 0.05, 100, 200, 10000)

    solver1 = pde.BlackScholesExplicitSolver(equation2)
    res1 = solver1.solve()
    # res2 = solver2.solve()

    print(res1.get_result())
    res1.plot()



if __name__ == "__main__":
    main()