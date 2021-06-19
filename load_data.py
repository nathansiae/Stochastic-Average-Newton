from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz
import numpy as np
from sklearn.datasets import load_svmlight_file


def get_data(data_path):
    """Once datasets are downloaded, load datasets in LibSVM format."""
    data = load_svmlight_file(data_path)
    return data[0], data[1]


def sparsity(data):
    # calculate data sparsity, i.e., number of zero entries / total entries in data matrix.
    # data, (num_samples, num_features)
    n, d = data.shape
    total_entries = n * d
    zeros_entries = np.sum(data == 0)
    return zeros_entries / total_entries

# ===============================
# These two functions to generate artificial data. Not used.
# =================================


def simu_linreg(x, n, std=1., corr=0.5):
    """Simulation for the least-squares problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    b = A.dot(x) + noise
    return A, b


def simu_logreg(x, n, std=1., corr=0.5):
    """Simulation for the logistic regression problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    A, b = simu_linreg(x, n, std=1., corr=corr)
    return A, np.sign(b)
# ======================================
