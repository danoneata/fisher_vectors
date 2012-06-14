import result_file_functions


def tuple_labels_to_list_labels(tuple_labels):
    """ Converts labels such as [(1, ), (3, )] to [1, 3]. """
    list_labels = []
    for label in tuple_labels:
        assert len(label) == 1, "The dataset should not be multilabel."
        list_labels.append(label[0])
    return list_labels


def average_precision(y_true, y_pred):
    """ Swaps arguments for Adrien's function, so it is compatible with
    sklearn.

    """
    return result_file_functions.get_ap(y_pred, y_true)
