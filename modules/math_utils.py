from numpy import dot
import numpy as np
from numpy.linalg import norm

import sklearn.metrics.pairwise
import torch

# normalized 0 .. 2 (0 ~ same)
def cosine_similarity_(a, b):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 2.0
    return 1.0 - dot(a, b)/(norm(a)*norm(b))


# normalized 0 .. 2 (0 ~ same)
def cosine_similarity(a, b, reduce=np.average, mode='numpy'):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 2.0
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=0)
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=0)
    n_jobs = 8 if a.shape[0] > 1 else 1

    if mode != 'numpy':
        a = torch.FloatTensor(a).to(mode)
        b = torch.FloatTensor(a).to(mode)

    with sklearn.config_context(working_memory=1024):
        result = np.zeros((0, ))
        for each in sklearn.metrics.pairwise.pairwise_distances_chunked(a, b, metric="cosine", n_jobs=n_jobs):
            result = np.concatenate((result, np.diag(each)), axis=0)

    if mode != 'numpy':
        result = result.to('cpu').numpy()

    if result.shape[0] == 1:
        return float(result[0])
    return float(reduce(result))


def cross_product(a, b):
    return normalize_vec(np.cross(a, b))


def normalize_vec(v):
    sum = np.sum(v**2)
    if sum == 0:
        return v
    v = v / np.sqrt(sum)
    return v