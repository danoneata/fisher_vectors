""" Collection of functions that convert the sufficient statistics into Fisher
vectors.

"""
from numpy import hstack, newaxis

from yael import yael


def sstats_to_fv(ss, gmm):
    """ Converts sufficient statistics into standard Fisher vectors. 
    
    Inputs
    ------
    ss: array [N, K + 2 * D * K]
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
          .reshape(D, K, order='F')[newaxis])    # 1xDxK
    sigma = (yael.fvec_to_numpy(gmm.sigma, K * D).
             reshape(D, K, order='F')[newaxis])  # 1xDxK

    # Get each part of the sufficient statistics.
    Q_sum = ss[:, :K]                            # NxK (then Nx1xK)
    Q_xx = ss[:, K:K + D * K]                    # NxKD
    Q_xx_2 = ss[:, K + D * K:K + 2 * D * K]      # NxKD

    # Compute the gradients.
    d_pi = Q_sum - pi
    d_mu = Q_xx - (Q_sum[:, newaxis] * mu).reshape(N, D * K, order='F')
    d_sigma = (
        - Q_xx_2
        - (Q_sum[:, newaxis] * mu ** 2).reshape(N, D * K, order='F')
        + (Q_sum[:, newaxis] * sigma).reshape(N, D * K, order='F')
        + 2 * Q_xx * mu.reshape(1, K * D, order='F'))

    # Glue everything together.
    fv = hstack((d_pi, d_mu, d_sigma))
    return fv 


def sstats_to_soft_bow(ss, gmm):
    """ Converts sufficient statistics into soft counts of bag-of-words, using
    the posterior probabilities.

    Inputs
    ------
    ss: array [N, K + 2 * D * K]
        Sufficient statistics.

    gmm: instance of the yael object, gmm
        Gaussian mixture model.

    Output
    ------
    soft_bow: array [N, K]
        Soft counts for each of the N sufficient statistics.

    """
    K = gmm.k
    D = gmm.d
    soft_bow = ss.reshape(-1, K + 2 * K * D)
    return soft_bow[:, :K]


def sstats_to_sfv(ss, gmm):
    """ Computes spatial Fisher vectors from spatial sufficient statistics.

    Inputs
    ------
    ss: array [N, K + 2 * 3 * K]
        Sufficient statistics for all of the N videos in the data set 
        (training or testing set). The sufficient statistics are obtained
        by applying the function _compute_spatial_statistics.

    gmm: instance of yael object gmm
        Gauassian mixture object.

    Output
    ------
    fv: array [N, 2 * 3 * K]
        Fisher vectors for each of the N videos in the data set. 

    """
    # Initialize parameters.
    K = gmm.k
    mm = self.mm[newaxis, :, newaxis]             # 1x3x1
    S = self.S[newaxis, :, newaxis]               # 1x3x1
    ss = ss.reshape(-1, K + 2 * 3 * K)
    N = ss.shape[0]

    # Get each part of the sufficient statistics.
    Q_sum = ss[:, :K]                             # NxK
    Q_ll = ss[:, K:K + 3 * K]                     # NxKD
    Q_ll_2 = ss[:, K + 3 * K:K + 2 * 3 * K]       # NxKD

    # Compute the gradients.
    d_mm = Q_ll - (Q_sum[:, newaxis] * mm).reshape(N, 3 * K, order='F')
    d_S = (
        - Q_ll_2
        - (Q_sum[:, newaxis] * mm ** 2).reshape(N, 3 * K, order='F')
        + (Q_sum[:, newaxis] * S).reshape(N, 3 * K, order='F')
        + 2 * Q_ll * tile(squeeze(mm)[newaxis], K))
    xx = hstack((d_mm, d_S))
    return xx
