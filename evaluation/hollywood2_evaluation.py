from ipdb import set_trace
import numpy as np

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

from .base_evaluation import BaseEvaluation
import result_file_functions


def average_precision(y_true, y_pred):
    """ Swaps arguments for Adrien's function, so it is compatible with
    sklearn.

    """
    return result_file_functions.get_ap(y_pred, y_true)


class Hollywood2Evaluation(BaseEvaluation):
    """ Evaluation procedure for the Hollywood2 dataset. Fits a one-vs-rest
    classifier for each class.

    """
    def __init__(self):
        pass

    def fit(self, Kxx, cx):
        """ Fits one-vs-rest classifiers for each class.

        Parameters
        ----------
        Kxx: array [n_samples, n_samples]
            Precomputed kernel matrix of the training data.

        cx: array [n_samples, n_classes]
            Labels matrix. Element cx[i, j] is 1 if sample `i` belongs to class
            `j`; otherwise it is -1. We preferred this encoding because some
            sample have multiple labels.

        """
        self.nr_classes = cx.shape[1]
        self.clf = []

        for ii in xrange(self.nr_classes):
            labels = cx[:, ii]

            # TODO Include class weights here.
            my_svm = svm.SVC(kernel='precomputed', probability=True)

            c_values = np.power(3.0, np.arange(-2, 8))
            tuned_parameters = [{'C': c_values}]

            splits = StratifiedShuffleSplit(labels, 3, test_size=0.3)
            self.clf.append(
                GridSearchCV(my_svm, tuned_parameters,
                             score_func=average_precision,
                             cv=splits, n_jobs=4))
            self.clf[ii].fit(Kxx, labels)

    def score(self, Kyx, cy):
        """ Returns the mean average precision score. """
        average_precisions = np.zeros(self.nr_classes)
        for ii in xrange(self.nr_classes):
            true_labels = cy[:, ii]
            predicted_values = self.clf[ii].predict_proba(Kyx)[:, 1]
            average_precisions[ii] = average_precision(
                true_labels, predicted_values)
        return np.mean(average_precisions)

    @classmethod
    def is_evaluation_for(cls, dataset_to_evaluate):
        if dataset_to_evaluate == 'hollywood2':
            return True
        else:
            return False
