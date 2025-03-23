import numpy as np

class HeatEquation:

    def __init__(self, length, x_nodes, time, t_nodes, k):
        self.__length = length
        self.__x_nodes = x_nodes
        self.__time = time
        self.__t_nodes = t_nodes
        self.__k = k
        self.__initial_temp = None
        self.__left_boundary_temp = None
        self.__right_boundary_temp = None

    def set_initial_temp(self, u0):
        self.__validate_callable(u0)
        if self.__length is None:
            raise RuntimeError("Rod length has not been initialised.")
        self.__initial_temp = u0
        self.__check_conditions()
        return self

    def set_left_boundary_temp(self, left):
        self.__validate_callable(left)
        self.__left_boundary_temp = left
        self.__check_conditions()
        return self

    def set_right_boundary_temp(self, right):
        self.__validate_callable(right)
        self.__right_boundary_temp = right
        self.__check_conditions()
        return self

    def __check_conditions(self):
        if self.__initial_temp is None:
            raise ValueError("Initial Temperature has not been initialised")

        if self.__left_boundary_temp is not None:
            err = np.abs(self.__left_boundary_temp(0) - self.__initial_temp(0))
            assert err < 1e-12, f"Left boundary condition at t=0 does not match the initial condition."

        if self.__right_boundary_temp is not None:
            err = np.abs(self.__right_boundary_temp(0) - self.__initial_temp(self.__length))
            assert err < 1e-12, f"Right boundary condition at t=0 does not match the initial condition."

    @staticmethod
    def __validate_callable(func):
        if not callable(func):
            raise ValueError("Temperature conditions must be a callable function")

    @staticmethod
    def generate_grid(value, nodes):
        return np.linspace(0, value, nodes)

    @property
    def length(self):
        return self.__length

    @property
    def time(self):
        return self.__time

    @property
    def x_nodes(self):
        return self.__x_nodes

    @property
    def t_nodes(self):
        return self.__t_nodes

    @property
    def k(self):
        return self.__k

    def get_initial_temp(self, x):
        return self.__initial_temp(x)

    def get_left_boundary(self, t):
        return self.__left_boundary_temp(t)

    def get_right_boundary(self, t):
        return self.__right_boundary_temp(t)

    @t_nodes.setter
    def t_nodes(self, nodes):
        self.__t_nodes = nodes
