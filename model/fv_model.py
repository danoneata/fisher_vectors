from ipdb import set_trace

import numpy as np
from numpy import dot
from numpy import newaxis
from numpy import hstack
from numpy import sqrt
from numpy import zeros

from .base_model import BaseModel

from yael import yael
from yael.yael import fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, GMM_FLAGS_W

from utils import standardize
from utils import power_normalize
from utils import compute_L2_normalization


class FVModel(BaseModel):
    # TODO Finish this.
    """ Fisher vectors model.

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
    def __init__(self, gmm):
        super(FVModel, self).__init__(gmm)
        self.is_spatial_model = False

    def __str__(self):
        # TODO
        ss = super(FVModel, self).__str__()
        return 'FV ' + ss

    @staticmethod
    def descs_to_sstats(xx, gmm):
        """ Converts the descriptors to sufficient statistics.

        Inputs
        ------
        xx: array [nr_descs, nr_dimensions]
            Data matrix containing the descriptors.

        gmm: yael.gmm instance
            Mixture of Gaussian object.

        Output
        ------
        sstats: array [nr_clusters + 2 * nr_clusters * nr_dimensions, ]
            Concatenation of the averaged posterior probabilities `Q_sum`, the
            first moment `Q_xx` and second-order moment `Q_xx_2`.

        """
        xx = np.atleast_2d(xx)
        N = xx.shape[0]
        K = gmm.k
        D = gmm.d
        # Compute posterior probabilities using yael.
        Q_yael = fvec_new(N * K)
        gmm_compute_p(N, numpy_to_fvec_ref(xx), gmm, Q_yael, GMM_FLAGS_W)
        Q = fvec_to_numpy(Q_yael, N * K).reshape(N, K)
        yael.free(Q_yael)
        # Compute statistics.
        sstats = np.zeros(K + 2 * K * D, dtype=np.float32)
        sstats[: K] = np.sum(Q, 0) / N                            # 1xK
        sstats[K: K + K * D] = dot(Q.T, xx).flatten() / N         # 1xKD
        sstats[K + K * D: K + 2 * K * D] = dot(
            Q.T, xx ** 2).flatten() / N                           # 1xKD
        return sstats

    @staticmethod
    def sstats_to_features(ss, gmm):
        """ Converts sufficient statistics into standard Fisher vectors.

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

    def compute_kernels(self, tr_file_paths, te_file_paths):
        """ Computes kernel matrices `Kxx` and `Kyx' from the given train and
        test files.

        """
        super(FVModel, self).compute_kernels(tr_file_paths, te_file_paths)
        self._init_normalizations()
        self._compute_kernels(tr_file_paths, te_file_paths)
        self._L2_normalize_kernels()

    def _init_normalizations(self):
        self.Zx = zeros(self.Nx)
        self.Zy = zeros(self.Ny)

    def _compute_kernels(self, train_paths, test_paths):
        for fn_train, fn_test in zip(train_paths, test_paths):
            # Process train set.
            ss = np.fromfile(fn_train, dtype=np.float32)

            xx = self.sstats_to_features(ss, self.gmm)
            xx, mu, sigma = standardize(xx)
            xx = power_normalize(xx, 0.5)
            self.Zx += compute_L2_normalization(xx)

            self.Kxx += dot(xx, xx.T)

            # Process test set.
            ss = np.fromfile(fn_test, dtype=np.float32)

            yy = self.sstats_to_features(ss, self.gmm)
            yy = standardize(yy, mu, sigma)[0]
            yy = power_normalize(yy, 0.5)
            self.Zy += compute_L2_normalization(yy)

            self.Kyx += dot(yy, xx.T)

    def _L2_normalize_kernels(self):
        # L2-normalize kernels.
        self.Kxx = self.Kxx / sqrt(self.Zx[:, newaxis] * self.Zx[newaxis])
        self.Kyx = self.Kyx / sqrt(self.Zy[:, newaxis] * self.Zx[newaxis])

    @classmethod
    def is_model_for(cls, type_model):
        if type_model == 'fv':
            return True
        else:
            return False
