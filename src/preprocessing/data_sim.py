import numpy as np
from numpy.random import multivariate_normal, uniform, random


def data_sim():
    no_features = 50
    no_samples = 300

    mean = uniform(low=-10.0, high=10.0, size=(no_features,))
    variance = uniform(low=-1.0, high=1.0, size=(no_features, no_features,))

    takes = multivariate_normal(mean, variance, no_samples)

    size = [10, 10, 10]  # remaining p - sum(size) is reference

    X_idx = [np.sum(size[:i]) for i in range(len(size) + 1)]
    X = [takes[:, int(l):int(r)] for l, r in zip(X_idx, X_idx[1:])]

    alpha = [uniform(low=-1.0, high=1.0, size=(s,)) for s in size]
    delta = random(len(size)) * 20

    val = [np.dot(x, a) for x, a in zip(X, alpha)]
    label = [v > d for v, d in zip(val, delta)]

    arr = np.empty(300, int)

    for i in range(300):
        if label[0][i]:
            arr[i] = 1
        elif label[1][i]:
            arr[i] = 2
        elif label[2][i]:
            arr[i] = 3
        else:
            arr[i] = 4

    return takes, arr
