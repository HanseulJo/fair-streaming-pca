import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from fair_streaming_pca import FairPCA

def compute_bandwidth(data_matrix, project_dim=10):
    pca_model = FairPCA() # for vanilla PCA
    pca_model.fit_offline(data_matrix, None, None, project_dim, constraint='vanilla')
    data_projected = pca_model.transform_low_dim(data_matrix)
    N = len(data_projected)
    dist_matrix = pairwise_distances(data_projected)
    distances = dist_matrix[np.triu_indices(N)]
    return np.sqrt(np.median(distances) / 2)

def mmd_rbf(X, Y, sigma=1.):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        sigma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    gamma = 1/sigma
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()