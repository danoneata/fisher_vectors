import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import raises

from dataset import SampID
from fisher_vectors.data import SstatsMap


class TestData():
    def setup(self):
        self.nr_dims = 10
        np.random.seed(1)
        self.data = SstatsMap('expected_results/test_data')
        # Generate some data.
        self.xx = np.random.randn(3, self.nr_dims)
        self.info = {
            'label': 4,
            'nr_slices': 3,
            'begin_frames': [1, 3, 10],
            'end_frames': [2, 9, 15],
            'nr_descs_per_slice': [10, 9, 13]}

    def test_write_data(self):
        self.data.write('test_1', self.xx[0])
        self.data.write('test_2', self.xx[1], info=self.info)
        self.data.write('test_3', self.xx[2], info=dict(label=10))

    def test_read_data(self):
        # Nose tests in random order.
        self.data.write('test_1', self.xx[0])
        self.data.write('test_2', self.xx[1], info=self.info)

        test_1_data = self.data.read('test_1')
        assert_array_almost_equal(self.xx[0], test_1_data)

        test_2_data = self.data.read('test_2')
        assert_array_almost_equal(self.xx[1], test_2_data)

        test_2_info = self.data.read_info('test_2')
        assert test_2_info == self.info

    @raises(Exception)
    def test_read_data_exception(self):
        self.data.read('file_that_does_not_exist')

    def test_check_data(self):
        self.data.write('test_1', self.xx)
        self.data.write('test_2', [])
        assert self.data.check(['test_1'], self.nr_dims, verbose=False) == True
        assert self.data.check(
            ['test_1'], self.nr_dims + 1, verbose=False) == False
        assert self.data.check(
            ['test_1', 'test_2'], self.nr_dims, verbose=False) == False
