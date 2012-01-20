# import numpy as np
from numpy import hstack, newaxis

from yael import yael
from .base_model import BaseModel


class FVModel(BaseModel):
    """ Implementation for Fisher vectors model.

    Extends the class BaseModel and implements the specific compute_features
    method.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    Examples
    --------

    """
    def __init__(self, K, grids):
        super(FVModel, self).__init__(K, grids)

    def __str__(self):
        # TODO Make this function appropriate.
        print 'Fisher vector model.'

    def compute_features(self, ss, gmm):
        """ Gets the statistics ss and the Gaussian mixture model object gmm
        and returns a NxD matrix containing the Fisher vectors.

        """
        K = gmm.k
        D = gmm.d
        ss = ss.reshape(-1, K + 2 * K * D)
        N = ss.shape[0]
        # Get parameters and reshape them.
        pi = yael.fvec_to_numpy(gmm.w, K)
        mu = (yael.fvec_to_numpy(gmm.mu, K * D)
              .reshape(D, K, order='F')[newaxis])
        sigma = (yael.fvec_to_numpy(gmm.sigma, K * D).
                 reshape(D, K, order='F')[newaxis])
        # Get each part of the sufficient statistics.
        Qs = ss[:, :K]
        Qxx = ss[:, K:K + D * K]
        Qxx2 = ss[:, K + D * K:K + 2 * D * K]
        # Compute derivatives.
        d_pi = Qs - pi
        d_mu = Qxx - (Qs[:, newaxis] * mu).reshape(N, D * K, order='F')
        d_sigma = (
            - Qxx2
            - (Qs[:, newaxis] * mu ** 2).reshape(N, D * K, order='F')
            + (Qs[:, newaxis] * sigma).reshape(N, D * K, order='F')
            + 2 * Qxx * mu.reshape(1, K * D, order='F'))
        xx = hstack((d_pi, d_mu, d_sigma))
        return xx

    @classmethod
    def is_model_for(cls, type_model):
        if type_model == 'fv':
            return True
        else:
            return False
