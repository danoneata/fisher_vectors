import result_file_functions
import numpy as np


def tuple_labels_to_list_labels(tuple_labels, positive_class=None):
    """ Converts labels such as [(1, ), (3, )] to [1, 3]. If positive class
    is given, the labels are also binarized.
    
    """
    list_labels = []
    for label in tuple_labels:
        assert len(label) == 1, "The dataset should not be multilabel."
        _label = label[0]
        if positive_class is not None:
            _label = 1 if _label == positive_class else -1
        list_labels.append(_label)
    return np.array(list_labels)


def average_precision(y_true, y_pred):
    """ Swaps arguments for Adrien's function, so it is compatible with
    sklearn.

    """
    return result_file_functions.get_ap(y_pred, y_true)


def calc_ap(labels, scores):
    """ Danila's way to compute mean average precision. """
    from bigimbaz.scripts.score import score_ap_from_ranks_1
    order = np.argsort(-np.array(scores))
    ranks = np.zeros_like(order)
    ranks[order] = np.arange(0, order.size)
    ranks = ranks[np.array(labels) > 0]
    ranks.sort()
    return score_ap_from_ranks_1(ranks, ranks.size)
