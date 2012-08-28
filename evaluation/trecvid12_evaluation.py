import numpy as np
from numpy import arange, array, ceil, Inf, mean
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import StratifiedShuffleSplit
from ipdb import set_trace

from .base_evaluation import BaseEvaluation
from utils import tuple_labels_to_list_labels
from utils import average_precision
#from utils import calc_ap as average_precision # Danila's metric.
import result_file_functions as rff


class TrecVid12Evaluation(BaseEvaluation):
    def __init__(self, **kwargs):
        self.null_class_idx = kwargs.get('null_class_idx', 0)
        self.scenario = kwargs.get('eval_type', 'trecvid11')

    def fit(self, Kxx, cx):
        if self.scenario == 'trecvid11':
            self.null_class_idx = 0
            self.fit_trecvid11(Kxx, cx)
        return self

    def score(self, Kyx, cy):
        if self.scenario == 'trecvid11':
            return self.score_trecvid11(Kyx, cy)
        
    def fit_trecvid11(self, Kxx, cx):
        """ Fits one-vs-rest classifier as for TrecVid 11. """
        self.nr_classes = len(set(cx))
        self.clf = []
        self.cx_idxs = []
        for ii, class_idx in enumerate(xrange(1, 16)):
            # Slice only the elements with index class_idx and null_class_idx.
            good_idxs = (cx == class_idx) | (cx == self.null_class_idx)
            self.cx_idxs.append(good_idxs)

            K_good_idxs = np.ix_(good_idxs, good_idxs)
            # Get a +1, -1 vector of labels.
            cx_ = map(lambda label: +1 if label != self.null_class_idx else -1,
                      cx[good_idxs])

            my_svm = svm.SVC(kernel='precomputed', probability=True,
                             class_weight='auto')  #{+1: 0.95, -1: 0.05})

            c_values = np.power(3.0, np.arange(-2, 8))
            #weights = [{+1: ww / (ww + 1), -1: 1 / (ww + 1)}
            #           for ww in np.power(2.0, np.arange(0,7))]
            #weights.append('auto')
            tuned_parameters = [{'C': c_values}]

            cx_ = np.array(cx_)
            splits = StratifiedShuffleSplit(cx_, 3, test_size=0.25,
                                            random_state=0)

            self.clf.append(
                GridSearchCV(my_svm, tuned_parameters,
                             score_func=average_precision,
                             cv=splits, n_jobs=4))
            self.clf[ii].fit(Kxx[K_good_idxs], cx_)
        return self

    def score_trecvid11(self, Kyx, cy):
        """ Returns the mean average precision score. """
        ap_scores = np.zeros(self.nr_classes - 1)
        for ii, class_idx in enumerate(xrange(1, 16)):
            good_idxs = (cy == class_idx) | (cy == self.null_class_idx)
            K_good_idxs = np.ix_(good_idxs, self.cx_idxs[ii])

            # Get a +1, -1 vector of labels.
            cy_ = map(
                lambda label: +1 if label == class_idx else -1,
                cy[good_idxs])
            # Predict.
            predicted_values = self.clf[ii].predict_proba(Kyx[K_good_idxs])[:, 1]
            ap_scores[ii] = average_precision(
                cy_, predicted_values)
            #print "%d sklearn %2.3f" % (ii, 100 * ap_scores[ii])
            #print 'Class #%d, test score: %2.3f' % (
            #    class_idx, 100 * ap_scores[ii])
            print '%d\t%2.3f' % (
                class_idx, 100 * ap_scores[ii])
        return np.mean(ap_scores) * 100

    @classmethod
    def is_evaluation_for(cls, dataset_to_evaluate):
        if dataset_to_evaluate == 'trecvid12':
            return True
        else:
            return False
