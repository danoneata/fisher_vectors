def tuple_labels_to_list_labels(tuple_labels):
    """ Converts labels such as [(1, ), (3, )] to [1, 3]. """
    list_labels = []
    for label in tuple_labels:
        assert len(label) == 1, "The dataset should not be multilabel."
        list_labels.append(label[0])
    return list_labels
