import numpy as np


def make_array_from_list(some_list):
    return np.array(some_list)


def make_array_from_number(num):
    return np.array(list(range(num, num+1)))


class NumpyBasics:
    def add_arrays(self, a, b):
        return np.add(a, b)

    def add_array_number(self, a, num):
        return np.add(a, make_array_from_number(num))

    def multiply_elementwise_arrays(self, a, b):
        return np.multiply(a, b)

    def dot_product_arrays(self, a, b):
        return np.dot(a, b)

    def dot_1d_array_2d_array(self, a, m):
        return m.dot(a)