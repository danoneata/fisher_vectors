from ipdb import set_trace
import numpy as np
from numpy import arange, array, ceil, Inf

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import zero_one_score
from utils import tuple_labels_to_list_labels

from .base_evaluation import BaseEvaluation


class KTHEvaluation(BaseEvaluation):
    """ Evaluation procedure for the KTH dataset. Fits a multiclass SVM in a
    one-vs-one style to the kernel matrix.
    
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, Kxx, cx):
        """ Fits SVM to kernel matrix and cross-validate the regularization 
        weight across a logarithmic scale.

        """
        cx = tuple_labels_to_list_labels(cx)
        my_svm = svm.SVC(kernel='precomputed')

        c_values = np.power(3.0, np.arange(-2, 8))
        tuned_parameters = [{'C': c_values}]

        splits = StratifiedShuffleSplit(cx, 3, test_size=0.3)

        self.clf = GridSearchCV(
            my_svm, tuned_parameters, score_func=zero_one_score,
            cv=splits, n_jobs=4)
        self.clf.fit(Kxx, cx)
        return self

    def score(self, Kyx, cy):
        """ Return the accuracy score / zero-one score. """
        cy = tuple_labels_to_list_labels(cy)
        return self.clf.score(Kyx, cy) * 100

    @classmethod
    def is_evaluation_for(cls, dataset_to_evaluate):
        if dataset_to_evaluate == 'kth':
            return True
        else:
            return False
