from ipdb import set_trace
import numpy as np
from numpy import arange, array, ceil, Inf, mean, ones, zeros
from sklearn import svm

from .base_evaluation import BaseEvaluation
import result_file_functions as rff


class SVMOneVsOne(BaseEvaluation):
    def __init__(self):
        pass

    def fit(self, Kxx, cx):
        self.nr_classes = cx.shape[1]
        self.clf = []
        N = Kxx.shape[0]
        for ii in xrange(self.nr_classes):
            #yy = self._select_labels(cx, ii)
            yy = cx[:,ii]
            nr_pos = len(yy[yy == 1])
            self.clf.append(svm.SVC(kernel='precomputed', probability=True))
            C = self._crossvalidate_C(Kxx, yy, ii)
            #weight = 1 * ones(N)
            #weight[yy == 1] *= N / nr_pos
            weight = 0.01 * ones(N)
            weight[yy == 1] *= 100 
            self.clf[ii].C = C
            self.clf[ii].fit(Kxx, yy, sample_weight=weight) 

    def score(self, Kyx, cy):
        average_precision = zeros(self.nr_classes)
        for ii in xrange(self.nr_classes):
            #yy = self._select_labels(cy, ii)
            yy = cy[:,ii]
            confidence_values = self.clf[ii].predict_proba(Kyx)[:, 1]
            average_precision[ii] = rff.get_ap(confidence_values, yy)
        return mean(average_precision)

    def _select_labels(self, cc, ii):
        cc = array(cc)
        yy = -1 * ones(len(cc))
        yy[cc == ii] = +1
        return yy

    def _crossvalidate_C(self, K, cc, idx_clf):
        # TODO Try to avoid duplication of some of this code.
        # 1. Split Gram matrix and labels into a training set and a validation
        # set.
        pp = 0.3  # Proportion of examples used for cross-validation.
        M, N = K.shape
        assert M == N, 'K is not Gram matrix.'
        classes = list(set(cc))
        nr_classes = len(classes)
        assert nr_classes == 2, 'Number of classes is less than two.'
        # Randomly pick a subset of the data for cross-validation, but enforce
        # to get a proportion of pp points from each of the two classes.
        idxs_0 = [ii for ii, ci in enumerate(cc) if ci == classes[0]]
        idxs_1 = [ii for ii, ci in enumerate(cc) if ci == classes[1]]
        rand_idxs_0 = np.random.permutation(idxs_0)
        rand_idxs_1 = np.random.permutation(idxs_1)
        P0 = ceil(pp * len(rand_idxs_0))
        P1 = ceil(pp * len(rand_idxs_1))
        cv_idxs = np.hstack((rand_idxs_0[:P0], rand_idxs_1[:P1]))
        tr_idxs = np.hstack((rand_idxs_0[P0:], rand_idxs_1[P1:]))
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
            self.clf[idx_clf].C = 3 ** log3c
            #weight = ones(len(tr_cc))
            #weight[tr_cc == 1] *= len(tr_cc) / len(tr_cc[tr_cc == 1])
            weight = 0.01 * ones(len(tr_cc))
            weight[tr_cc == 1] *= 100 
            self.clf[idx_clf].fit(tr_K, tr_cc, sample_weight=weight) 
            confidence_values = self.clf[idx_clf].predict_proba(cv_K)[:,1]
            score = rff.get_ap(confidence_values, cv_cc)
            if score >= best_score:
                best_score = score
                best_C = self.clf[idx_clf].C
        return best_C

    @classmethod
    def is_evaluation_for(cls, type_evaluation):
        if type_evaluation == 'svm_one_vs_one':
            return True
        else:
            return False
