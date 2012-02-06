# TODO Write unit tests using either py.test, unittest or nose!
import os
import pytest
import numpy as np
from numpy.testing import assert_equal

from dataset import Dataset
from fisher_vectors.features import DescriptorProcessor

K = 50
d = 64
dataset = Dataset('kth', ip_type='harris.hoghof')

def test_init():
    dp = DescriptorProcessor(dataset, K)
    assert dp._exist_descriptors()

def test_load_subsample_descriptors():
    dp = DescriptorProcessor(dataset, K)
    descriptors = dp._load_subsample_descriptors()
    (N,D) = descriptors.shape
    assert N < 2e5

def test_merge_tmp_statistics():
    dp = DescriptorProcessor(dataset, K)
    # dp.compute_statistics(1)
    # dp._merge_tmp_statistics('train')
    ss = dataset.get_data('train')[0]
    N = len(ss)
    all_fn = os.path.join(dataset.FEAT_DIR, 'statistics', 'train_1_1_1_0.dat')
    yy_fn = os.path.join(dataset.FEAT_DIR, 'statistics.tmp', '%s_1_1_1_0.dat' %
                         ss[1])
    all_train = np.fromfile(all_fn).reshape(N, 2 * K * d + K)
    xx = all_train[1,:]
    yy = np.fromfile(yy_fn)
    assert_equal(xx, yy)

