import numpy as np
import scipy.linalg as la
from numpy.random import multivariate_normal, uniform, random, randint


def data_sim(
        no_features=50,
        no_samples=600,
        delta_factor=1,
        starts=[0, 10, 20],
        sizes=[10, 10, 10],
        random_labels=False):
    """Generate simulated data with four classes
    where given feature ranges condition class belonging"""

    mean = uniform(low=-1.0, high=1.0, size=(no_features,))
    eigs = uniform(low=1.0, high=10.0, size=(no_features,))
    s = np.diag(eigs)
    q, _ = la.qr(uniform(low=10.0, high=100.0, size=(no_features, no_features)))
    variance = q.T @ s @ q

    takes = multivariate_normal(mean, variance, no_samples)

    X = [takes[:, int(l):int(l + r)] for l, r in zip(starts, sizes)]

    alpha = [uniform(low=-1.0, high=1.0, size=(s,)) for s in sizes]
    delta = random(len(sizes)) * delta_factor

    val = [np.dot(x, a) for x, a in zip(X, alpha)]
    label = [v > d for v, d in zip(val, delta)]

    arr = np.empty(no_samples, int)

    for i in range(no_samples):
        if label[0][i] == True:
            arr[i] = 1
        elif label[1][i] == True:
            arr[i] = 2
        elif label[2][i] == True:
            arr[i] = 3
        else:
            arr[i] = 4

    if random_labels:
        arr = randint(1, 5, no_samples)

    return takes, arr
