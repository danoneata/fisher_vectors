import os
import numpy as np


DESCS_LEN = {
    'mbh': 192,
    'hog': 96,
    'hof': 108,
    'hoghof': 96 + 108,
    'all': 96 + 108 + 192}


def load_subsample_descriptors(dataset):
    """ Returns a NxD dimensional matrix representing a subsample of the
    descriptors. The subset files are either subset.siftgeo or subset.dat.
    Their location is induced from dataset.
    
    Note: the temporal information is discarded.
    
    """
    filename =  os.path.join(dataset.FEAT_DIR, 'subset')
    if os.path.exists(filename + '.siftgeo'):
        from video_vwgeo import read_video_points_from_siftgeo
        siftgeos = read_video_points_from_siftgeo(filename + '.siftgeo')
        N = len(siftgeos)
        D = len(siftgeos[0][1])
        # Put the descriptors in a numpy matrix.
        descriptors = zeros((N, D), dtype=np.float32)
        for ii, siftgeo in enumerate(siftgeos):
            descriptors[ii] = siftgeo[1]
    elif os.path.exists(filename + '.dat'):
        # TODO Make a function that given the FEAT_TYPE returns the descriptor
        # length.
        descriptors = np.fromfile(
            filename + '.dat', dtype=np.float32).reshape(
                (-1, DESCS_LEN['mbh'] + 3))
        # Discard temporal information.
        descriptors = descriptors[:, 3:]
    else:
        raise IOError
    return descriptors
