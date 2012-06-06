from ipdb import set_trace
import numpy as np
from numpy import exp, float64, mean, newaxis, sum

from .base_model import BaseModel
from ekovof.dense.dense_distances import m2m_chisquare as chi_square_distance
from yael import yael
from yael.yael import fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, GMM_FLAGS_W


class BOWModel(BaseModel):
    """ Implementation for bag of words model.

    """
    def __init__(self, K, grids):
        super(BOWModel, self).__init__(K, grids)
        self.is_spatial_model = False

    def __str__(self):
        ss = super(BOWModel, self).__str__()
        return 'BOW ' + ss

    @staticmethod
    def _compute_statistics(xx, gmm):
        """ Converts the descriptors to sufficient statistics.
        
        Inputs
        ------
        xx: array [nr_descs, nr_dimensions]
            Data matrix containing the descriptors.

        gmm: yael.gmm instance
            Mixture of Gaussian object.

        Output
        ------
        Q_sum: array [nr_clusters, ]
            Averaged posterior probabilities.

        """
        N = xx.shape[0]
        # Compute posterior probabilities using yael.
        Q_yael = fvec_new(N * self.K)
        gmm_compute_p(N, numpy_to_fvec_ref(xx), gmm, Q_yael, GMM_FLAGS_W)
        Q = fvec_to_numpy(Q_yael, N * self.K).reshape(N, self.K)
        yael.free(Q_yael)
        # Compute statistics.
        Q_sum = sum(Q, 0) / N                     # 1xK
        return np.array(Q_sum, dtype=np.float32)

    def compute_kernels(self, dataset):
        super(BOWModel, self)._init_kernels(dataset)
        self._compute_distances(dataset, self._compute_features)
        self._kernelize_distances()

    def _init_kernels(self, dataset):
        super(BOWModel, self)._init_kernels(dataset)
        self.Zx = 0.
        self.Zy = 0.

    def _compute_distances(self, dataset, _compute_features, prefix=''):
        train_paths, test_paths = self._get_statistics_paths(dataset, prefix)
        for fn_train, fn_test in zip(train_paths, test_paths):
            # Process train set.
            ss = np.fromfile(fn_train, dtype=np.float32)
            xx = np.array(_compute_features(ss, self.gmm), dtype=float64)
            self.Kxx += chi_square_distance(xx, xx)
            # Process test set.
            ss = np.fromfile(fn_test, dtype=np.float32)
            yy = np.array(_compute_features(ss, self.gmm), dtype=float64)
            self.Kyx += chi_square_distance(yy, xx)

    def _kernelize_distances(self):
        self.Kxx = exp(- self.Kxx / mean(self.Kxx)) 
        self.Kyx = exp(- self.Kyx / mean(self.Kyx)) 

    def _compute_features(self, ss, gmm):
        K = gmm.k
        D = gmm.d
        ss = ss.reshape(-1, K + 2 * K * D)
        return ss[:, :K]

    def _chi_square_distance(self, xx, yy):  # Deprecated. Using Adrien's version.
        """ Computes the chi square distance betweent two sets of "histograms",
        i.e., array of positive numbers that sum up to one.

        Parameters
        ----------
        xx: array [M, D], required
            Data points.

        yy: array [N, D], required
            Data points.

        Attributes
        ----------
        dist: array [M, N]
            Chi square distances between each pair of points.

        """ 
        xx = xx[:, newaxis]
        yy = yy[newaxis]
        ss = xx + yy
        ss[ss < 1e-7] = 1.  # To avoid division by zero or small numbers.
        return sum((xx - yy) ** 2 / ss, 2)

    @classmethod
    def is_model_for(cls, type_model):
        if type_model == 'bow':
            return True
        else:
            return False
