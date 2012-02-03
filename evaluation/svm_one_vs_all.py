import numpy as np
from numpy import arange, array, ceil, Inf
from sklearn import svm
from ipdb import set_trace

from .base_evaluation import BaseEvaluation


class SVMOneVsAll(BaseEvaluation):
    def __init__(self):
        pass

    def fit(self, Kxx, cx):
        self.clf = svm.SVC(kernel='precomputed')
        C = self._crossvalidate_C(Kxx, cx)
        self.clf.C = C
        self.clf.fit(Kxx, cx)

    def score(self, Kyx, cy):
        return self.clf.score(Kyx, cy)

    def _crossvalidate_C(self, K, cc):
        # 1. Split Gram matrix and labels into a training set and a validation
        # set.
        pp = 0.3  # Proportion of examples used for cross-validation.
        M, N = K.shape
        assert M == N, 'K is not Gram matrix.'
        classes = list(set(cc))
        nr_classes = len(classes)
        assert nr_classes >= 2, 'Number of classes is less than two.'
        # Randomly pick a subset of the data for cross-validation.
        rand_idxs = np.random.permutation(arange(N))
        P = ceil(pp * N)
        cv_idxs = rand_idxs[:P]
        tr_idxs = rand_idxs[P:]
        # Get indices in numpy format.
        cv_ix_ = np.ix_(cv_idxs, tr_idxs)
        tr_ix_ = np.ix_(tr_idxs, tr_idxs)
        # Slice Gram matrix K.
        cv_K = K[cv_ix_]
        tr_K = K[tr_ix_]
        # Get corresponding labels.
        cc = array(cc)
        cv_cc = cc[cv_idxs]
        tr_cc = cc[tr_idxs]
        # 2. Try different values for the regularization term C and pick the 
        # one that yields the best score on the cross-validation set.
        log3cs = arange(-2, 8)  # Vary C on an exponantional scale.
        best_score = - Inf
        best_C = 0
        for log3c in log3cs:
            self.clf.C = 3 ** log3c
            self.clf.fit(tr_K, tr_cc)
            score = self.clf.score(cv_K, cv_cc)
            if score >= best_score:
                best_score = score
                best_C = self.clf.C
        return best_C

    @classmethod
    def is_evaluation_for(cls, type_evaluation):
        if type_evaluation == 'svm_one_vs_all':
            return True
        else:
            return False
