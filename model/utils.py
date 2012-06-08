import numpy as np


def standardize(xx, mu=None, sigma=None):
    """ If the mu and sigma parameters are None, returns the standardized data,
    i.e., the zero mean and unit variance data, along with the corresponding
    mean and variance that were used for this standardization. Otherwise---if
    mu and sigma are given---, fron the data xx is substracted mu and then xx
    is multiplied by sigma on each dimension.

    Inputs
    ------
    xx: array [N, D]
        Data.

    mu: array [D], default None
        Mean of the data along the columns.

    sigma: array [D], default None
        Variance on each dimension.

    Outputs
    -------
    xx: array [N, D]
        Standardized data.

    mu: array [D]
        Computed or given mean. 

    sigma: array [D]
        Computed or given variance.

    """
    if xx.ndim != 2:
        raise ValueError, "Input array must be two dimensional."

    if mu is None or sigma is None:
        mu = np.mean(xx, 0)
        sigma = np.std(xx - mu, 0)

    if np.min(sigma) == 0:
        raise ValueError, "There is zero variance on some dimension."

    return (xx - mu) / sigma, mu, sigma


def power_normalize(xx, alpha):
    """ Computes a alpha-power normalization for the matrix xx. """
    return np.sign(xx) * np.abs(xx) ** alpha


def compute_L2_normalization(xx):
    """ Computes the L2 norm along the rows, i.e. for each example.

    Input
    -----
    xx: array [N, D]
        Data.

    Output
    ------
    Z: array [N]
        Normalization terms.

    """
    return np.sum(xx * xx, 1)


def L2_normalize(xx):
    """ L2-normalizes each row of the data xx.

    Input
    -----
    xx: array [N, D]
        Data.

    Output
    ------
    yy: array [N, D]
        L2-normlized data.

    """
    Zx = compute_L2_normalization(xx)
    return xx / np.sqrt(Zx[:, np.newaxis])
