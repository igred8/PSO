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

def ackley_path_function(*args):
    # takes k-dimensional vector
    # continuous, multi-model
    # global minimum: x_i = 0 
    # rescale onto [-32, 32]
    vector = np.array(args).squeeze()
    a = 20.
    b = 0.2
    c = np.pi
    n = float(len(vector))
    result = - a * np.exp( - b * np.sqrt(np.sum(vector**2) / n ) ) - np.exp( np.sum(np.cos(c * vector)) / n ) + a + np.exp(1.)
    return result
