from collections import defaultdict
import os

from ipdb import set_trace
from numpy import prod, zeros


class BaseModel(object):
    def __init__(self, K, grids):
        self.K = K
        self.grids = grids
        self.base_fn = 'statistics_k_%d' % K

    def __str__(self):
        return str(self.K) + ' ' + str(self.grids)

    def fit(self, dataset, evaluation):
        self.compute_kernels(dataset)
        self.cx = dataset.get_data('train')[1]
        self.cy = dataset.get_data('test')[1]
        self.evaluation = evaluation
        self.evaluation.fit(self.Kxx, self.cx)

    def predict(self):
        """ Only for plotting at the moment. """
        return self.evaluation.predict(self.Kyx, self.cy)

    def score(self):
        return self.evaluation.score(self.Kyx, self.cy)

    def get_kernels(self):
        return self.Kxx, self.Kyx

    def _compute_features(self):
        pass

    def _init_kernels(self, dataset):
        self.Nx = len(dataset.get_data('train')[0])
        self.Ny = len(dataset.get_data('test')[0])
        self.Kxx = zeros((self.Nx, self.Nx))
        self.Kyx = zeros((self.Ny, self.Nx))
        self.gmm = self._get_gmm(dataset)

    def _get_gmm(self, dataset):
        from yael.yael import gmm_read
        fn_gmm = os.path.join(dataset.FEAT_DIR, 'gmm', 'gmm_%d' % self.K)
        with open(fn_gmm, 'r') as ff:
            gmm = gmm_read(ff)
            return gmm

    def _get_filenames(self, prefix=''):
        train_files = []
        test_files = []
        bare_fn = os.path.join(self.base_fn, prefix + '%s_%d_%d_%d_%d.dat')
        for grid in self.grids:
            for ii in xrange(prod(grid)):
                train_files.append(
                    bare_fn % ('train', grid[0], grid[1], grid[2], ii))
                test_files.append(
                    bare_fn % ('test', grid[0], grid[1], grid[2], ii))
        return train_files, test_files

    def _get_statistics_paths(self, dataset, prefix=''):
        """ Create two lists containing the paths to the files containing the 
        statistics.

        """
        train_files, test_files = self._get_filenames(prefix)
        train_paths = [os.path.join(dataset.FEAT_DIR, train_file)
                       for train_file in train_files]
        test_paths = [os.path.join(dataset.FEAT_DIR, test_file)
                      for test_file in test_files]
        return train_paths, test_paths

    @classmethod
    def is_model_for(cls, type_model):
        return False

    class __metaclass__(type):
        """ Define a new attribute of class BaseModel that has a dictionary 
        with all the classes that inherit from BaseModel.

        """
        __inheritors__ = defaultdict(list)
        def __new__(meta, name, bases, dct):
            cls = type.__new__(meta, name, bases, dct)
            for base in cls.mro()[1:-1]:
                meta.__inheritors__[base].append(cls)
            return cls
