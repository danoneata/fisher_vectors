import result_file_functions
import numpy as np


def tuple_labels_to_list_labels(tuple_labels, positive_class=None):
    """ Converts labels such as [(1, ), (3, )] to [1, 3]. If positive class
    is given, the labels are also binarized.
    
    """
    list_labels = []
    for label in tuple_labels:
        if positive_class is not None:
            _label = 1 if positive_class in label else -1
        list_labels.append(_label)
    return np.array(list_labels)


def average_precision(y_true, y_pred):
    """ Swaps arguments for Adrien's function, so it is compatible with
    sklearn.

    """
    return result_file_functions.get_ap(y_pred, y_true)


def detection_cost_rate(y_true, y_pred):
    return - compute_dcr(y_pred, y_true)


def calc_ap(labels, scores):
    """ Danila's way to compute mean average precision. """
    from bigimbaz.scripts.score import score_ap_from_ranks_1
    order = np.argsort(-np.array(scores))
    ranks = np.zeros_like(order)
    ranks[order] = np.arange(0, order.size)
    ranks = ranks[np.array(labels) > 0]
    ranks.sort()
    return score_ap_from_ranks_1(ranks, ranks.size)


def compute_dcr(confvals, gt):
    Wmiss = 1.0
    Wfa = 12.49999

    res = zip(confvals, gt)
    res.sort()
    res.reverse()
    tp_ranks = [(rank, confval)
                for rank, (confval, istp) in enumerate(res)
                if istp > 0]

    ntot = len(confvals)
    npos = len(tp_ranks)
    dcr_tab = [Wmiss]

    for i, (rank, confval) in enumerate(tp_ranks):
        # consider results >= confval
        nres = rank + 1      # nb of results
        ntp = i + 1          # nb of tp
        nfa = nres - ntp     # nb of false alarms
        nmiss = npos - ntp   # nb of missed
        Pfa = nfa / float(ntot - npos)
        Pmiss = nmiss / float(npos)
        dcr = Wmiss * Pmiss + Wfa * Pfa
        dcr_tab.append(dcr)

    return min(dcr_tab)
