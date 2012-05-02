""" Collection of functions that convert the sufficient statistics into Fisher
vectors.

"""
from numpy import hstack, newaxis

from yael import yael


def sstats_to_fv(sstats, gmm):
    """ Gets the statistics sstats and the Gaussian mixture model object gmm
    and returns a NxD matrix containing the Fisher vectors.

    """
    K = gmm.k
    D = gmm.d
    ss = sstats.reshape(-1, K + 2 * K * D)
    N = ss.shape[0]

    # Get parameters and reshape them.
    pi = yael.fvec_to_numpy(gmm.w, K)            # 1xK
    mu = (yael.fvec_to_numpy(gmm.mu, K * D)
          .reshape(D, K, order='F')[newaxis])    # 1xDxK
    sigma = (yael.fvec_to_numpy(gmm.sigma, K * D).
             reshape(D, K, order='F')[newaxis])  # 1xDxK

    # Get each part of the sufficient statistics.
    Q_sum = ss[:, :K]                            # NxK (then Nx1xK)
    Q_xx = ss[:, K:K + D * K]                    # NxKD
    Q_xx_2 = ss[:, K + D * K:K + 2 * D * K]      # NxKD

    # Compute derivatives.
    d_pi = Q_sum - pi
    d_mu = Q_xx - (Q_sum[:, newaxis] * mu).reshape(N, D * K, order='F')
    d_sigma = (
        - Q_xx_2
        - (Q_sum[:, newaxis] * mu ** 2).reshape(N, D * K, order='F')
        + (Q_sum[:, newaxis] * sigma).reshape(N, D * K, order='F')
        + 2 * Q_xx * mu.reshape(1, K * D, order='F'))

    # Glue everything together.
    xx = hstack((d_pi, d_mu, d_sigma))
    return xx
