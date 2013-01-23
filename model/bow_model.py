from ipdb import set_trace
import numpy as np
from numpy import exp, float64, mean, newaxis, sum

from .base_model import BaseModel
from ekovof.dense.dense_distances import m2m_chisquare as chi_square_distance
from yael import yael
from yael.yael import fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, GMM_FLAGS_W


class BOWModel(BaseModel):
    # TODO
    """ Implementation for bag of words model.

    """
    def __init__(self, gmm):
        super(BOWModel, self).__init__(gmm)
        self.is_spatial_model = False

    def __str__(self):
        # TODO
        ss = super(BOWModel, self).__str__()
        return 'BOW ' + ss

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
        Q_sum: array [nr_clusters, ]
            Averaged posterior probabilities.

        """
        K = gmm.k
        N = xx.shape[0]
        # Compute posterior probabilities using yael.
        Q_yael = fvec_new(N * K)
        gmm_compute_p(N, numpy_to_fvec_ref(xx), gmm, Q_yael, GMM_FLAGS_W)
        Q = fvec_to_numpy(Q_yael, N * K).reshape(N, K)
        yael.free(Q_yael)
        # Compute statistics.
        Q_sum = sum(Q, 0) / N                     # 1xK
        return np.array(Q_sum, dtype=np.float32)

    @staticmethod
    def sstats_to_features(ss, gmm, nr_sstats):
        """ Converts sufficient statistics into soft counts of bag-of-words,
        using the posterior probabilities.

        Inputs
        ------
        ss: array [N * (K + 2 * D * K), ]
            Sufficient statistics.

        gmm: instance of the yael object, gmm
            Gaussian mixture model.

        nr_sstats: int
            Number of sufficient statistics, N.

        Output
        ------
        soft_bow: array [N, K]
            Soft counts for each of the N sufficient statistics.

        Note
        ----
        We need the number of sufficient statistics because, the features can
        be obtain from either the full sufficient statistics (computed with
        `descs_to_sstats` from the `fv` model) or they can consist only of the
        posterior probabilities (computed with `descs_to_sstats` from `self`).

        """
        K = gmm.k
        soft_bow = ss.reshape(nr_sstats, -1)
        return soft_bow[:, :K]

    def compute_kernels(self, tr_file_paths, te_file_paths):
        super(BOWModel, self).compute_kernels(tr_file_paths, te_file_paths)
        self._compute_distances(tr_file_paths, te_file_paths)
        self._kernelize_distances()

    def _compute_distances(self, train_paths, test_paths):
        for fn_train, fn_test in zip(train_paths, test_paths):
            # Process train set.
            ss = np.fromfile(fn_train, dtype=np.float32)
            xx = np.array(self.sstats_to_features(ss, self.gmm, self.Nx),
                          dtype=float64)
            self.Kxx += chi_square_distance(xx, xx)

            # Process test set.
            ss = np.fromfile(fn_test, dtype=np.float32)
            yy = np.array(self.sstats_to_features(ss, self.gmm, self.Ny),
                          dtype=float64)
            self.Kyx += chi_square_distance(yy, xx)

    def _kernelize_distances(self):
        self.Kxx = exp(- self.Kxx / mean(self.Kxx)) 
        self.Kyx = exp(- self.Kyx / mean(self.Kyx))  # Normalize by mean(Kxx) ?

    @classmethod
    def is_model_for(cls, type_model):
        if type_model == 'bow':
            return True
        else:
            return False
