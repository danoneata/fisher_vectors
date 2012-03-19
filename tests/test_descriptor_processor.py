import ipdb
import os
import pytest
import numpy as np
import numpy.testing 
import subprocess

from dataset import Dataset
from fisher_vectors.features import DescriptorProcessor
from fisher_vectors.model import Model

K = 50
d = 64

def test_init():
    """ Tests simple initialization of a descriptor processor instance. """
    dataset = Dataset('kth')
    model = Model('fv', K)
    dp = DescriptorProcessor(dataset, model)

 
def test_subsample_descriptors():
    """ Tests loading or computing the subset of descriptors. """
    dataset = Dataset('kth')
    model = Model('fv', K)
    dp = DescriptorProcessor(dataset, model)
    try:
        descriptors = dp._load_subsample_descriptors()
    except IOError:
        dp._compute_subsample_descriptors(2e5)
        descritpors = dp._load_subsample_descriptors()
    (N, D) = descriptors.shape
    assert 1.9e5 <= N <= 2.1e5


def test_compute_statistics_from_video():
    """ Tests the consistency of results between the two approaches of
    computing the sufficient statistics: (i) from the .siftgeo files or (ii)
    directly from the video.
    
    """
    dataset = Dataset('kth', ip_type='dense5.track15mbh')
    model = Model('fv', K)
    dp = DescriptorProcessor(dataset, model)
    dp.exist_statistics = False
    
    pca = dp.load_pca()
    gmm = dp.load_gmm()

    samples, labels = dataset.get_data('train')
    sample = samples[130]  # Pick a random sample.
    
    siftgeo_fn = os.path.join(dataset.FEAT_DIR, 'features',
                              str(sample) + '.siftgeo')
    file_moved = False
    try:
        # If .siftgeo exists move it temporarily.
        siftgeo_ff = open(siftgeo_fn, 'r')
        siftgeo_ff.close()
        subprocess.Popen(['mv', siftgeo_fn, siftgeo_fn + '.old']).wait()
        file_moved = True
    except IOError:
        pass
    # Compute new .siftgeo file. We recompute it because there are some
    # difference when computing the dense trajectories.
    adrien_dense_tracks = ('/home/clear/gaidon/progs/' +
                           'dense_trajectory/release64/FeatTrack')
    my_dense_tracks = 'densetracks'
    new_dense_tracks = '/home/lear/oneata/tmp/dense_trajectory/release64/FeatTrack'
    infile = os.path.join( dataset.SRC_DIR, sample.movie + dataset.SRC_EXT)
    outfile = siftgeo_fn
    subprocess.Popen([new_dense_tracks, infile, outfile, '15',
                      '5', str(sample.bf), str(sample.ef), 'mbh']).wait()
    # Computing the sufficient statistics for the case (i).
    dp.compute_statistics_worker([sample], (1, 1, 1), pca, gmm)
    fn = os.path.join(dataset.FEAT_DIR, 'statistics_k_%d' % K, 'stats.tmp',
                      sample + '_1_1_1_0.dat')
    desc_1 = np.fromfile(fn, dtype=np.float32)

    # Move sufficient statistics. 
    subprocess.Popen(['mv', fn, fn + '.old']).wait()

    # Computing the sufficient statistics for the case (ii).
    dp.compute_statistics_from_video_worker([sample], (1, 1, 1), pca, gmm, 0, 0)
    #ipdb.set_trace()
    desc_2 = np.fromfile(fn, dtype=np.float32)

    # Do the cleaning.
    subprocess.Popen(['rm', '-f', fn, fn + '.old']).wait()
    if file_moved:
        # Revert.
        #subprocess.Popen(['rm', '-f', siftgeo_fn]).wait()
        #subprocess.Popen(['mv', siftgeo_fn + '.old', siftgeo_fn]).wait()
        pass

    print desc_1
    print desc_2
    numpy.testing.assert_array_almost_equal(desc_1, desc_2, 3)

test_compute_statistics_from_video()
