from collections import defaultdict

from ipdb import set_trace
import numpy as np


class BaseModel(object):
    def __init__(self, gmm):
        self.gmm = gmm

    def __str__(self):
        # TODO
        return str(self.K) + ' ' + str(self.grids)

    def compute_kernels(self, tr_files_paths, te_files_paths, **kwargs):
        len_sstats = self.gmm.k + 2 * self.gmm.k * self.gmm.d

        self.Nx = np.fromfile(
            tr_files_paths[0], dtype=np.float32).reshape(
                (-1, len_sstats)).shape[0]
        self.Ny = np.fromfile(
            te_files_paths[0], dtype=np.float32).reshape(
                (-1, len_sstats)).shape[0]

        self.Kxx = np.zeros((self.Nx, self.Nx))
        self.Kyx = np.zeros((self.Ny, self.Nx))

    def get_kernels(self):
        return self.Kxx, self.Kyx

    @staticmethod
    def sstats_to_features():
        pass

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
