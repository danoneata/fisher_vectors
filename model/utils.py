import numpy as np
import warnings

from yael import yael


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
        warnings.warn("At least one dimension of the data has zero variance.")
        sigma[sigma == 0] = 1.

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


def sstats_to_sqrt_features(ss, gmm):
    """ Converts sufficient statistics into an approximated form of sqrt
    Fisher vectors.

    Inputs
    ------
    ss: array [N * (K + 2 * D * K), ]
        Sufficient statistics.

    gmm: instance of the yael object, gmm
        Gaussian mixture model.

    Output
    ------
    fv: array [N, K + 2 * D * K]
        Fisher vectors for each of the N sufficient statistics.

    """
    K = gmm.k
    D = gmm.d
    ss = ss.reshape(-1, K + 2 * K * D)
    N = ss.shape[0]

    # Get parameters and reshape them.
    pi = yael.fvec_to_numpy(gmm.w, K)            # 1xK
    mu = (yael.fvec_to_numpy(gmm.mu, K * D)
          .reshape(D, K, order='F')[np.newaxis])    # 1xDxK
    sigma = (yael.fvec_to_numpy(gmm.sigma, K * D).
             reshape(D, K, order='F')[np.newaxis])  # 1xDxK

    # Get each part of the sufficient statistics.
    Q_sum = ss[:, :K]                            # NxK (then Nx1xK)
    Q_xx = ss[:, K:K + D * K]                    # NxKD
    Q_xx_2 = ss[:, K + D * K:K + 2 * D * K]      # NxKD

    # Sqrt the soft counts.
    sqrtQ = np.sqrt(Q_sum)

    # Compute the gradients.
    d_pi = sqrtQ
    d_mu = Q_xx - (Q_sum[:, np.newaxis] * mu).reshape(N, D * K, order='F')
    d_mu = d_mu / np.repeat(sqrtQ, D)
    d_sigma = (
        - Q_xx_2
        - (Q_sum[:, np.newaxis] * mu ** 2).reshape(N, D * K, order='F')
        + (Q_sum[:, np.newaxis] * sigma).reshape(N, D * K, order='F')
        + 2 * Q_xx * mu.reshape(1, K * D, order='F')) / np.repeat(sqrtQ, D)

    # Glue everything together.
    fv = np.hstack((d_pi, d_mu, d_sigma))
    return fv
