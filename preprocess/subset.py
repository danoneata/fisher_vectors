import os
import numpy as np

from constants import DESCS_LEN


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
        descriptors = np.zeros((N, D), dtype=np.float32)
        for ii, siftgeo in enumerate(siftgeos):
            descriptors[ii] = siftgeo[1]
    elif os.path.exists(filename + '.dat'):
        # TODO Make a function that given the FEAT_TYPE returns the descriptor
        # length.
        descs_len = DESCS_LEN[dataset.FTYPE.split('.')[-1]] + (
            0 if 'mfcc' in dataset.FTYPE else 3)
        descriptors = np.fromfile(
            filename + '.dat', dtype=np.float32).reshape(-1, descs_len)
        # Discard temporal information.
        descriptors = descriptors[:, 3:]
    else:
        raise IOError
    return descriptors
