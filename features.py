from __future__ import division
from collections import defaultdict
from itertools import product
import os

import cPickle as cp
from numpy import digitize, dot, float32, hstack, linspace
from numpy import sum, prod, vstack, zeros
from sklearn.decomposition import PCA
# from ipdb import set_trace

from dataset import SampID
from vidbase.vidplayer import get_video_infos
from video_vwgeo import read_video_points_from_siftgeo
from yael.yael import count_cpu, fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, gmm_learn, gmm_read, gmm_write
from yael.yael import GMM_FLAGS_W
from yael import yael

verbose = True  # Global variable used for printing messages.


class FeaturesWorker:
    """ Feature processing class.

    This class contains functions and utilities that are used for the low-level
    and mid-level processing of the features. These include: computing PCA on a
    subset of descriptors, building a vocabulary, computing statistics that can
    be latter used in Fisher vectors construction.

    Parameters
    ----------
    dataset: instance of class Dataset, required

    K: int, required
        The number of words used for the vocabulary.

    grid: tuple, default (1,1,1)
        A tuple containing 3 values that represents the splitting grid for
        spatial pyramid matching tehnique.

    nr_pca_comps: int, default 64
        Number of principal components used for dimensionality reduction of the
        features.

    Attributes
    ----------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, dataset, K, grid=(1, 1, 1), nr_pca_comps=64):
        self.dataset = dataset
        self.K = K
        self.grid = grid
        self.nr_pca_comps = nr_pca_comps
        self.NR_CPUS = count_cpu()
        # Set filenames that will be used later on.
        self.fn_pca = os.path.join(self.dataset.FEAT_DIR, 'pca',
                                   'pca_%d.pkl' % self.nr_pca_comps)
        self.fn_gmm = os.path.join(self.dataset.FEAT_DIR, 'gmm',
                                   'gmm_%d' % self.K)
        # TODO Assert low-level features exist.

    def __str__(self):
        pass

    def foo(self):
        """ Does all! Is it nice enough? I might not want to do this. """
        try:
            descriptors = self.load_subsampled_descriptors()
        except IOError:
            # TODO Check thrown exception is correct.
            descriptors = self.subsample_descriptors(2e5)
            self.save_subsampled_descriptors(descriptors)

        try:
            pca = self.load_pca()
        except IOError:
            pca = self.compute_pca()
            self.save_pca(pca)

        try:
            self.load_gmm(pca)
        except IOError:
            gmm = self.compute_gmm()
            self.save_gmm(gmm)

    def compute_pca(self, descriptors):
        """ Computes PCA on a subset of nr_samples of descriptors. """
        pca = PCA(n_components=self.nr_pca_comps)
        pca.fit(descriptors)
        return pca

    def save_pca(self, pca):
        with open(self.fn_pca, 'w') as ff:
            cp.dump(pca, ff)

    def load_pca(self):
        """ Loads PCA object from file using cPickle. """
        with open(self.fn_pca, 'r') as ff:
            pca = cp.load(ff)
            return pca

    def compute_gmm(self, pca, nr_iter=100, nr_threads=None,
                    seed=1, nr_redo=4, nr_samples=2e5):
        """ Computes GMM using yael functions. """
        if nr_threads is None:
            nr_threads = self.NR_CPUS
        descriptors = self._get_subsample_descriptors(nr_samples)
        if pca:
            descriptors = pca.transform(descriptors)
        N, D = descriptors.shape
        gmm = gmm_learn(D, N, self.K, nr_iter, numpy_to_fvec_ref(descriptors),
                        nr_threads, seed, GMM_FLAGS_W)
        return gmm

    def save_gmm(self, gmm):
        with open(self.fn_gmm, 'w') as ff:
            gmm_write(gmm, ff)

    def load_gmm(self):
        """ Loads GMM object from file using yael. """
        with open(self.fn_gmm, 'r') as ff:
            gmm = gmm_read(ff)
            return gmm

    def compute_statistics(self, nr_processes=None):
        """ Computes sufficient statistics needed for the bag-of-words or
        Fisher vector model.

        """
        if nr_processes is None:
            nr_processes = self.NR_CPUS

        train_samples = self.dataset.get_samples('train')[0]
        test_samples = self.dataset.get_samples('test')[0]
        samples = list(set(train_samples + test_samples))

        pca = self.load_pca()
        gmm = self.load_gmm()

        if nr_processes > 1:
            import multiprocessing as mp
            nr_samples_per_process = len(samples) // nr_processes
            for ii in xrange(nr_processes):
                process = mp.Process(
                    target=self.compute_statistics_worker,
                    args=(self, samples[ii * nr_samples_per_process],
                          self.grid, pca, gmm))
                process.start()
        else:
            self.compute_statistics_worker(self, samples, self.grid)

    def compute_statistics_worker(self, samples, grid, pca, gmm):
        """ Worker function for computing the sufficient statistics. It takes
        each sample, read the points and their locations and computes the
        sufficient statistics for the points in each individual bin. The bins
        are specified by grid.

        """
        for sample in samples:
            sample_id = SampID(sample)
            # Prepare descriptors: select according to the grid and apply PCA.
            siftgeo = self.dataset.read_siftgeo_points(sample_id)
            video_infos = get_video_infos(
                os.path.join(self.dataset.PREFIX, 'videos',
                             sample_id.movie + '.avi'))
            W, H = video_infos['img_size']
            bag = self._process_descriptors(siftgeo, pca, grid, (W, H),
                                            (sample_id.bf, sample_id.ef))
            all_x = range(1, grid[0] + 1)
            all_y = range(1, grid[1] + 1)
            all_t = range(1, grid[2] + 1)
            ss = zeros((prod(grid),
                        self.K + 2 * self.K * self.nr_pca_comps))
            for ii, bin in enumerate(product(all_x, all_y, all_t)):
                fn = os.path.join(
                    self.dataset.FEAT_DIR, 'statistics', '%s_%d_%d_%d_%d.dat'
                    % (sample, grid[0], grid[1], grid[2], ii))
                with open(fn, 'w') as ff:
                    if bag[bin]:
                        idx = (bin[0] - 1 + grid[0] * (bin[1] - 1)
                               + grid[0] * grid[1] * (bin[2] - 1))
                        ss[idx] = self._compute_statistics(
                            vstack(bag[bin]), gmm)
                    ss.tofile(ff)

    def _process_descriptors(self, siftgeo, pca, grid, dimensions, duration):
        """ Groups the points in different bins using the gridding specified
        by grid. The returned results is a dictionary that has a key the bin
        number on each of the three dimensions x, y and t.

        """
        W, H = dimensions
        t_init, t_final = duration
        # Create equally spaced bins.
        bins_x = linspace(1, W, grid[0] + 1)
        bins_y = linspace(1, H, grid[1] + 1)
        bins_t = linspace(t_init, t_final + 1, grid[2] + 1)
        bag = defaultdict(list)
        for ss in siftgeo:
            xx = pca.transform(ss[1])
            id_x = digitize([ss[0]['x']], bins_x)
            id_y = digitize([ss[0]['y']], bins_y)
            id_t = digitize([ss[0]['t']], bins_t)
            bag[(id_x[0], id_y[0], id_t[0])].append(xx)
            assert (1 <= id_x <= grid[0] and
                    1 <= id_y <= grid[1] and
                    1 <= id_t <= grid[2])
        return bag

    def _compute_statistics(self, xx, gmm):
        """ Worker function for statistics computations. Takes as input a NxD 
        data matrix xx and the gmm object. Returns the corresponding statistics
        ---a vector of length D * (2 * K + 1).

        """
        N = xx.shape[0]
        # Compute posterior probabilities using yael.
        Q_yael = fvec_new(N * self.K)
        gmm_compute_p(N, numpy_to_fvec_ref(xx), gmm, Q_yael, GMM_FLAGS_W)
        Q = fvec_to_numpy(Q_yael, N * self.K).reshape(N, self.K)
        yael.free(Q_yael)
        # Compute statistics.
        Q_sum = sum(Q, 0)
        Q_xx = dot(Q.T, xx).flatten()
        Q_xx_2 = dot(Q.T, xx ** 2).flatten()
        return hstack((Q_sum, Q_xx, Q_xx_2))

    def load_sample_statistics(self, sample):
        samples = (self.dataset.get_data('train')[0] +
                   self.dataset.get_data('test')[0])
        assert sample in samples
        pass

    def _get_subsample_descriptors(self, nr_samples):
        if verbose:
            print 'Sub-sampling descriptors...'
        bash_cmd = ('/home/clear/oneata/scripts/bash_functions/' +
                    'run_subsample.sh %s %s %s' %
                    (self.dataset.DATASET, self.dataset.FTYPE,
                     str(nr_samples)))
        os.system(bash_cmd)
        # Maybe put this in an utility function.
        siftgeos = read_video_points_from_siftgeo(self.fn_subsample)
        N = len(siftgeos)
        D = len(siftgeos[0][1])
        descriptors = zeros((N, D), dtype=float32)
        for ii, siftgeo in enumerate(siftgeos):
            descriptors[ii] = siftgeo[1]
        return descriptors
