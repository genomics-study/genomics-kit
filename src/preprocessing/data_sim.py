import numpy as np
from numpy.random import multivariate_normal, rand, uniform, random
from numpy import tril


def data_sim():

    p = 60
    N = 300

    mean = uniform(low=-10.0, high=10.0, size=(p,))
    variance = uniform(low=-1.0, high=1.0, size=(p, p,))

    takes = multivariate_normal(mean, variance, N)

    size_A1 = 10
    size_A2 = 10
    size_A3 = 10
    size_A4 = 20

    alfa_A1 = uniform(low=-1.0, high=1.0, size=(size_A1,))
    delta_A1 = random(1)[0]

    alfa_A2 = uniform(low=-1.0, high=1.0, size=(size_A2,))
    delta_A2 = random(1)[0]

    vals = np.dot(takes[:, :10], alfa_A1)

    # TODO now filter that vals above delta_A2 and we have labels

    # TODO Classification

    # TODO Repeat 10k
