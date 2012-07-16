from nose.tools import raises
import numpy as np
from numpy.testing import assert_allclose
from ipdb import set_trace

from fisher_vectors.model import utils


class TestUtils():
    def setup(self):
        self.D = 100

    @raises(ValueError)
    def test_standardize_one_dimensional(self):
        yy, mu, sigma = utils.standardize(np.random.rand(self.D))
