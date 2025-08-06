import numpy as np

class HeatEquation2D:
    def __init__(self, time, t_nodes, k, xlength, x_nodes, ylength=None, y_nodes=None):
        self.__time = time
        self.__t_nodes = t_nodes
        self.__k = k
        self.__xlength = xlength
        self.__x_nodes = x_nodes
        self.__ylength = ylength if ylength is not None else xlength
        self.__y_nodes = y_nodes if y_nodes is not None else x_nodes
        # Initialize boundary conditions to None
        self.__initial_temp = None
        self.__left_boundary = None
        self.__right_boundary = None
        self.__top_boundary = None
        self.__bottom_boundary = None
        
    def set_initial_temp(self, u0):
        self.__initial_temp = u0

    def set_left_boundary_temp(self, left):
        self.__left_boundary = left

    def set_right_boundary_temp(self, right):
        self.__right_boundary = right

    def set_top_boundary_temp(self, top):
        self.__top_boundary = top

    def set_bottom_boundary_temp(self, bottom):
        self.__bottom_boundary = bottom

    @property
    def xlength(self):
        return self.__xlength

    @property
    def ylength(self):
        return self.__ylength

    @property
    def x_nodes(self):
        return self.__x_nodes

    @property
    def y_nodes(self):
        return self.__y_nodes

    @property
    def time(self):
        return self.__time

    @property
    def t_nodes(self):
        return self.__t_nodes

    @property
    def k(self):
        return self.__k
    
    @property
    def initial_temp(self):
        return self.__initial_temp

    @property
    def left_boundary(self):
        return self.__left_boundary

    @property
    def right_boundary(self):
        return self.__right_boundary

    @property
    def top_boundary(self):
        return self.__top_boundary

    @property
    def bottom_boundary(self):
        return self.__bottom_boundary