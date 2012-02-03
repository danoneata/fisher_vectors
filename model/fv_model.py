import numpy as np
from numpy import abs, dot, hstack, mean, newaxis, sign, sum, sqrt, std, zeros

from .base_model import BaseModel
from yael import yael
from yael.yael import fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, GMM_FLAGS_W

class FVModel(BaseModel):
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
    def __init__(self, K, grids):
        super(FVModel, self).__init__(K, grids)
        self.is_spatial_model = False

    def __str__(self):
        ss = super(FVModel, self).__str__()
        return 'FV ' + ss

    def _compute_statistics(self, xx, gmm):
        """ Worker function for statistics computations. Takes as input a NxD
        data matrix xx and the gmm object. Returns the corresponding statistics
        ---a vector of length D * (2 * K + 1). Using these statistics we can 
        then easily compute the Fisher vectors or a soft bag-of-words histogram

        """
        N = xx.shape[0]
        # Compute posterior probabilities using yael.
        Q_yael = fvec_new(N * self.K)
        gmm_compute_p(N, numpy_to_fvec_ref(xx), gmm, Q_yael, GMM_FLAGS_W)
        Q = fvec_to_numpy(Q_yael, N * self.K).reshape(N, self.K)
        yael.free(Q_yael)
        # Compute statistics.
        Q_sum = sum(Q, 0) / N                     # 1xK
        Q_xx = dot(Q.T, xx).flatten() / N         # 1xKD
        Q_xx_2 = dot(Q.T, xx ** 2).flatten() / N  # 1xKD
        return np.array(hstack((Q_sum, Q_xx, Q_xx_2)), dtype=np.float32)

    def compute_kernels(self, dataset):
        self._init_kernels(dataset)
        self._compute_kernels(dataset, self._compute_features)
        self._L2_normalize_kernels()
        return self.Kxx, self.Kyx
    
    def _init_kernels(self, dataset):
        super(FVModel, self)._init_kernels(dataset)
        self.Zx = zeros(self.Nx)
        self.Zy = zeros(self.Ny)

    def _compute_kernels(self, dataset, _compute_features, prefix=''):
        """ Separated function to reuse it more easily. """
        train_paths, test_paths = self._get_statistics_paths(dataset, prefix)
        for fn_train, fn_test in zip(train_paths, test_paths):
            # Process train set.
            ss = np.fromfile(fn_train, dtype=np.float32)
            xx = _compute_features(ss, self.gmm)
            xx, mu, sigma = self._standardize(xx)
            xx = self._power_normalize(xx, 0.5)
            #xx = self._L2_normalize(xx)
            self.Zx += self._compute_L2_normalization(xx)
            self.Kxx += dot(xx, xx.T)
            # Process test set.
            ss = np.fromfile(fn_test, dtype=np.float32)
            yy = _compute_features(ss, self.gmm)
            yy = self._standardize(yy, mu, sigma)[0]
            yy = self._power_normalize(yy, 0.5)
            #yy = self._L2_normalize(yy)
            self.Zy += self._compute_L2_normalization(yy)
            self.Kyx += dot(yy, xx.T)

    @classmethod
    def _compute_features(cls, ss, gmm, fn=None):
        """ Gets the statistics ss and the Gaussian mixture model object gmm
        and returns a NxD matrix containing the Fisher vectors.

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
        # Compute derivatives.
        d_pi = Q_sum - pi
        d_mu = Q_xx - (Q_sum[:, newaxis] * mu).reshape(N, D * K, order='F')
        d_sigma = (
            - Q_xx_2
            - (Q_sum[:, newaxis] * mu ** 2).reshape(N, D * K, order='F')
            + (Q_sum[:, newaxis] * sigma).reshape(N, D * K, order='F')
            + 2 * Q_xx * mu.reshape(1, K * D, order='F'))
        xx = hstack((d_pi, d_mu, d_sigma))
        return xx

    def _standardize(self, xx, mu=None, sigma=None):
        """ Returns the standardized data, i.e., zero mean and unit variance
        data, and the corresponding mean and standard deviation.

        """
        if mu is None or sigma is None:
            mu = mean(xx, 0)
            sigma = std(xx - mu, 0)
        return (xx - mu) / sigma, mu, sigma

    def _power_normalize(self, xx, alpha):
        """ Computes a alpha-power normalization for the matrix xx. """
        return sign(xx) * abs(xx) ** alpha

    def _compute_L2_normalization(self, xx):
        """ Input: a NxD dimensional matrix. Returns the sum over lines, hence
        a N dimensional vector.

        """
        return sum(xx * xx, 1)

    def _L2_normalize(self, xx):
        Zx = self._compute_L2_normalization(xx)
        return xx / sqrt(Zx[:, newaxis])

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
