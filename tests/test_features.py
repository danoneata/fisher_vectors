import os

from ipdb import set_trace
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from nose.tools import assert_almost_equal

from video_vwgeo import read_video_points_from_siftgeo

from fisher_vectors.features import get_slice_number
from fisher_vectors.features import parse_ip_type
from fisher_vectors.features import read_descriptors_from_video


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


class TestParseIpType():
    def setup(self):
        self.tests = ['dense5.track15mbh', 'dense5.track20mbh']
        self.results = [('5', '15', 'mbh'), ('5', '20', 'mbh')]

    def test_parse_ip_type(self):
        for test, result in zip(self.tests, self.results):
            assert result == parse_ip_type(test)


def siftgeo_to_matrix(siftgeo):
    """ Converts a siftgeo to a matrix whose rows are the descriptors, the
    first three values being the location (x, y, t) and the next the actual
    descriptor.

    """
    nr_descs = len(siftgeo)
    len_desc = len(siftgeo[0][0]) + len(siftgeo[0][1])
    xx = np.empty((nr_descs, len_desc), dtype=np.float32)
    for ii, desc in enumerate(siftgeo):
        xx[ii, :] = np.hstack((desc[0]['x'], desc[0]['y'],
                               desc[0]['t'], desc[1]))
    return xx


class TestReadDescriptorsFromVideo():
    def setup(self):
        # Parameters are given in the following order.
        #   infile, ip_type, nr_descriptors, begin_frames, end_frames and
        #   result_file.
        infile = os.path.expanduser(
            '~/data/kth/videos/person01_boxing_d1.avi')
        ip_type = 'dense5.track15mbh'
        nr_descriptors_1 = 1
        nr_descriptors_2 = 100
        begin_frames = [1]
        end_frames = [95]
        result_file = 'expected_results/person01_boxing_d1-frames-1-95.siftgeo'
        self.parameters = [
            [infile, ip_type, nr_descriptors_1, begin_frames, end_frames,
             result_file], 
            [infile, ip_type, nr_descriptors_2, begin_frames, end_frames,
             result_file]
        ] 

    def test_read_descriptors_from_video(self):
        for params in self.parameters:
            infile, ip_type, nr_descriptors, begin_frames, end_frames, result_file = params
            siftgeo = []
            for chunk in read_descriptors_from_video(
                infile, nr_descriptors=nr_descriptors, ip_type=ip_type,
                begin_frames=begin_frames, end_frames=end_frames):

                if nr_descriptors == 1:
                    assert chunk.shape[0] == 1
                else:
                    assert chunk.shape[0] > 1

                siftgeo.append(chunk)

            result = np.vstack(siftgeo)
            expected_result = siftgeo_to_matrix(
                read_video_points_from_siftgeo(result_file))
            assert_allclose(result[:, 3:], expected_result[:, 3:], rtol=1e-05)
