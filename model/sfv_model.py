from ipdb import set_trace
import numpy as np
from numpy import dot, hstack, newaxis, squeeze, tile

from .base_model import BaseModel
from yael import yael
from yael.yael import fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, GMM_FLAGS_W

class SFVModel(BaseModel):
    """ Spatial Fisher vector model.

    Implements functions for computing spatial statistics and spatial features
    for the Spatial Fisher model with C = 1 Gaussian, as in 3.2 from (Krapac 
    et al, 2011).

    Parameters
    ----------
    K: int, required
        The number of words in the vocabulary.

    grids: list of tuples, optional, default [(1, 1, 1)]
        The grids that are used to split the video for pyramid matching.

    Attributes
    ----------
    mm: array [1, 3]
        The mean of the Gaussian.

    S: array [1, 3]
        The diagonal of the covariance matrix for the Gaussian.

    Notes
    -----
    This class should not be instantiated. FVSFVModel inherits from it and 
    uses the ``spatial'' methods.

    """
    def __init__(self, K, grids):
        super(SFVModel, self).__init__(K, grids)
        self.mm = np.array([0.5, 0.5, 0.5])
        self.S = np.array([1. / 12, 1. / 12, 1. / 12])

    def __str__(self):
        ss = super(SFVModel, self).__str__()
        return 'FV-SFV ' + ss

    def _compute_spatial_statistics(self, xx, ll, gmm):
        """ Computes spatial statistics from descriptors and their position.

        Inputs
        ------
        xx: array [N, D], required
            N D-dimensional descriptors from an video (usually, after they are
            processed with PCA).

        ll: array [N, 3], required
            Descriptor locations in an image; on each row, we have the triplet
            (x, y, t).

        gmm: instance of yael object gmm
            Gauassian mixture object.

        Output
        ------
        ss: array [1, K + 2 * 3 * k]
            Sufficient statistics in the form of a vector that concatenates 
            (i) the sum of posteriors, (ii) an expected value of the locations 
            ll under the posterior distribution Q and (iii) the second-order
            moment of the locations ll under the posterior distribution Q.

        """
        N = ll.shape[0] 
        # Compute posterior probabilities using yael.
        Q_yael = fvec_new(N * self.K)
        gmm_compute_p(N, numpy_to_fvec_ref(xx), gmm, Q_yael, GMM_FLAGS_W)
        Q = fvec_to_numpy(Q_yael, N * self.K).reshape(N, self.K)
        yael.free(Q_yael)
        # Compute statistics.
        Q_sum = sum(Q, 0) / N                     # 1xK
        Q_ll = dot(Q.T, ll).flatten() / N         # 1x3K
        Q_ll_2 = dot(Q.T, ll ** 2).flatten() / N  # 1x3K 
        return np.array(hstack((Q_sum, Q_ll, Q_ll_2)), dtype=np.float32)

    def _compute_spatial_features(self, ss, gmm):
        """ Computes spatial features from spatial sufficient statistics.

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
        K = gmm.k
        mm = self.mm[newaxis, :, newaxis]             # 1x3x1
        S = self.S[newaxis, :, newaxis]               # 1x3x1
        ss = ss.reshape(-1, K + 2 * 3 * K)
        N = ss.shape[0]
        Q_sum = ss[:, :K]                             # NxK
        Q_ll = ss[:, K:K + 3 * K]                     # NxKD
        Q_ll_2 = ss[:, K + 3 * K:K + 2 * 3 * K]       # NxKD
        d_mm = Q_ll - (Q_sum[:, newaxis] * mm).reshape(N, 3 * K, order='F')
        d_S = (
            - Q_ll_2
            - (Q_sum[:, newaxis] * mm ** 2).reshape(N, 3 * K, order='F')
            + (Q_sum[:, newaxis] * S).reshape(N, 3 * K, order='F')
            + 2 * Q_ll * tile(squeeze(mm)[newaxis], K))
        xx = hstack((d_mm, d_S))
        return xx

    @classmethod
    def is_model_for(cls, type_model):
        return False
