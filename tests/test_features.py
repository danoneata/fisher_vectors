import numpy as np
from nose.tools import raises
from nose.tools import with_setup

from fisher_vectors.features import get_slice_number


class TestGetSliceNumber():
    def setup(self):
        self.begin_frames = np.array([1., 10., 120.])
        self.end_frames = np.array([5., 15., 150.])
        self.current_frames = np.array([1., 12., 123., 144.])
        self.expected_results = np.array([0, 1, 2, 2])
        self.known_fail = np.array([1000.])

    def test_get_slice_number(self):
        for current_frame, expected_result in zip(self.current_frames,
                                                  self.expected_results):
            assert expected_result == get_slice_number(
                current_frame, self.begin_frames, self.end_frames) 

    @raises(Exception)
    def test_get_slice_number_known_fails(self):
        get_slice_number(self.known_fail, self.begin_frames, self.end_frames)
