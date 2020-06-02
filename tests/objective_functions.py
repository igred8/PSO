import numpy as np


def rosenbrock_function(*args):
    # takes k-dimensional vector as either k inputs or array of length k
    # continuous, convex, unimodal, narrow valley
    # global minimum: x_i = 0 for all i in range(k)
    # rescale onto [-2, 2]
    vector = np.array(args).squeeze()
    result = 0
    for index, element in enumerate(vector[:-1]):
        result += 100 * (vector[index + 1] - element**2)**2  + (1 - element)**2
    return result


def rastrigin_function(*args):
    # takes k-dimensional vector as either k inputs or array of length k
    # continuous, multi-modal, regular distribution of minima
    # global minimum: x_i = 0 for all i in range(k)
    # Domain [-5, 5]
    vector = np.array(args).squeeze()
    result = 10. * len(vector)
    for index, element in enumerate(vector):
        result += element**2 - 10 * np.cos(2 * np.pi * element)
    return result

from pathos.multiprocessing import Pool, cpu_count

pool = Pool(4)
x = np.random.random([10, 2])
