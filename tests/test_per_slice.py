import os

from nose.tools import raises
from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_allclose
from ipdb import set_trace
import cPickle as pickle

from dataset import Dataset
from fisher_vectors.per_slice.features import merge_class_given_dataset
from fisher_vectors.per_slice.discriminative_detection import aggregate
from fisher_vectors.per_slice.discriminative_detection import _normalize
from fisher_vectors.per_slice.discriminative_detection import chunker


class TestPerSliceFeatures():
    def setup(self):
        self.dataset = Dataset('trecvid11_small', nr_cluster=128,
                               suffix='.small.per_slice')
        self.K = 128
        self.D = 64
        self.sstats_len = self.K + 2 * self.K * self.D

    def test_merge_class_given_dataset(self):
        out_folder = '/tmp/'
        out_fn = os.path.join(out_folder, 'train_class_0.dat')
        labels_fn = os.path.join(out_folder, 'labels_train_class_0.info')

        nr_pos = 30
        nr_neg = 60

        merge_class_given_dataset(
            self.dataset, 0, out_folder=out_folder, nr_positive_samples=nr_pos,
            nr_negative_samples=nr_neg) 

        # Load sufficient statistics.
        sstats = np.fromfile(out_fn, dtype=np.float32).reshape(
            (-1, self.sstats_len))
        labels = pickle.load(open(labels_fn, 'rb'))

        # Check if the number is accurate.
        assert sstats.shape[0] == len(labels), "Size mismatch."


class TestPerSliceData():
    def setup(self):
        self.scores = np.array([10., 10., 20., 40., 40.])
        self.limits = np.array([0, 2, 5])
        self.norm_scores = np.array([0.5, 0.5, 0.2, 0.4, 0.4])
        self.sstats = np.array([
            [ 0.55104342,  0.25347533,  0.29457834],
            [ 0.97736164,  0.25547196,  0.77425607],
            [ 0.89445082,  0.08111661,  0.94853142],
            [ 0.55631951,  0.36082249,  0.6745448 ],
            [ 0.23283677,  0.6410189 ,  0.17864344]])
        self.aggregated_sstats = np.array([
            [ 0.76420253,  0.25447365,  0.53441721],
            [ 0.49455268,  0.41695988,  0.53098158]])
        self.limits_long = np.array([0, 6, 10, 12, 15])
        self.split_limits_ling = [np.array([0, 6, 10]), np.array([12, 15])]

    def test_normalize_scores(self):
        results = _normalize(self.scores, self.limits)
        assert_allclose(results, self.norm_scores)

    def test_aggregate_sstats(self):
        results = aggregate(self.sstats, self.norm_scores,
                                       self.limits)
        assert_allclose(results, self.aggregated_sstats)

    def test_chunk_normalization(self):
        result = np.zeros_like(self.scores)
        for limits in chunker(self.limits, 2):
            low = limits[0]
            high = limits[-1]
            result[low: high] = _normalize(
                self.scores[low: high], limits - low)
        assert_allclose(result, self.norm_scores)
