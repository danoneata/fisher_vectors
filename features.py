from __future__ import division
from collections import defaultdict
from itertools import product
import os
import sys

import cPickle as cp
import numpy as np
from numpy import digitize, dot, hstack, linspace
from numpy import sum, prod, vstack, zeros
from sklearn.decomposition import PCA
from ipdb import set_trace

from dataset import SampID
from vidbase.vidplayer import get_video_infos
from video_vwgeo import read_video_points_from_siftgeo
from yael.yael import count_cpu, fvec_new, fvec_to_numpy, numpy_to_fvec_ref
from yael.yael import gmm_compute_p, gmm_learn, gmm_read, gmm_write
from yael.yael import GMM_FLAGS_W
from yael import yael

verbose = True  # Global variable used for printing messages.


class DescriptorProcessor:
    """ Descriptor processing class.

    This class contains functions and utilities that are used for the low-level
    and mid-level processing of the descriptors. These include: computing PCA
    on a subset of descriptors, building a vocabulary, computing statistics
    that can be later used in Fisher vectors construction.

    Parameters
    ----------
    dataset: instance of class Dataset, required

    K: int, required
        The number of words used for the vocabulary.

    grid: tuple, optional, default (1, 1, 1)
        A tuple containing 3 values that represents the splitting grid for
        spatial pyramid matching tehnique.

    nr_pca_comps: int, optional, default 64
        Number of principal components used for dimensionality reduction of the
        features.

    Attributes
    ----------
    fn_subsample: string
        Path to the file containing the subsample of the descriptors.

    fn_pca: string
        Path to the file containing the PCA object for the given configuration.

    fn_gmm: string
        Path to the file containing the GMM object for the given configuration.

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.K = self.model.K
        self.grid = self.model.grids[0]
        self.NR_CPUS = count_cpu()
        # Set filenames that will be used later on.
        self.nr_pca_comps = 64
        self.nr_samples = 2e5
        self.fn_subsample = os.path.join(self.dataset.FEAT_DIR,
                                         'subset.siftgeo')
        self.fn_pca = os.path.join(self.dataset.FEAT_DIR, 'pca',
                                   'pca_%d.pkl' % self.nr_pca_comps)
        self.fn_gmm = os.path.join(self.dataset.FEAT_DIR, 'gmm',
                                   'gmm_%d' % self.K)
        self.bare_fn = '%s_%d_%d_%d_%d.dat'
        self.spatial_bare_fn = 'spatial_%s_%d_%d_%d_%d.dat'
        self.root_path = os.path.join(self.dataset.FEAT_DIR, 'statistics_k_%d' % self.K)
        # Create path for temporary files.
        self.temp_path = os.path.join(self.root_path, 'stats.tmp')

    def __str__(self):
        pass

    def foo(self):
        """ Does all! Is it nice enough? I might not want to do this. """
        try:
            descriptors = self.load_subsampled_descriptors()
        except IOError:
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
        descriptors = self._load_subsample_descriptors(nr_samples)
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
        # Check if files containing the statistics exist.
        self.exist_statistics = self._exist_statistics()
        self.exist_spatial_statistics = self._exist_statistics('spatial_')
        # If they do exist, just exit.
        if self.exist_statistics and (self.exist_spatial_statistics or not self.model.is_spatial_model):
            print 'Statistic files already exist. Chickening out...'
            return

        if nr_processes is None:
            nr_processes = self.NR_CPUS

        # Create temporary folder where the statistics will be stored.
        # If foldeer already exists ignore.
        try:
            os.mkdir(self.temp_path)
        except OSError:
            pass

        # Get the unique set of labels. We do this for data sets that are 
        # multi-labeled and a specific sample appears multiple times in the 
        # training or testing list.
        train_samples = self.dataset.get_data('train')[0]
        test_samples = self.dataset.get_data('test')[0]
        samples = list(set(train_samples + test_samples))

        pca = self.load_pca()
        gmm = self.load_gmm()

        if nr_processes > 1:
            import multiprocessing as mp
            processes = []
            nr_samples_per_process = len(samples) // nr_processes + 1
            for ii in xrange(nr_processes):
                process = mp.Process(
                    target=self.compute_statistics_worker,
                    args=(samples[ii * nr_samples_per_process :
                                  (ii + 1) * nr_samples_per_process],
                          self.grid, pca, gmm))
                processes.append(process)
                process.start()
            # Wait for jobs to finish.
            for process in processes:
                process.join()
        else:
            # We use this special case, because it makes possible to debug.
            self.compute_statistics_worker(samples, self.grid, pca, gmm)

    def merge_statistics(self):
        self._merge_tmp_statistics('train')
        self._merge_tmp_statistics('test')
        if self.model.is_spatial_model:
            self._merge_tmp_statistics('train', 'spatial_')
            self._merge_tmp_statistics('test', 'spatial_')

    def remove_statistics(self):
        self._remove_tmp_statistics()
        if self.model.is_spatial_model:
            self._remove_tmp_statistics('spatial_') 
        os.rmdir(self.temp_path)

    def compute_statistics_worker(self, samples, grid, pca, gmm):
        """ Worker function for computing the sufficient statistics. It takes
        each sample, read the points and their locations and computes the
        sufficient statistics for the points in each individual bin. The bins
        are specified by grid.

        """
        for sample in samples:
            sample_id = SampID(sample)
            # Prepare descriptors: select according to the grid and apply PCA.
            siftgeo = read_video_points_from_siftgeo(os.path.join(
                self.dataset.FEAT_DIR, 'features', sample + '.siftgeo'))
            # TODO Use function get_video_resolution from Dataset
            video_infos = get_video_infos(
                os.path.join(self.dataset.PREFIX, 'videos',
                             sample_id.movie + '.avi'))
            W, H = video_infos['img_size']
            # Filter descriptors into multiple bags according to their location.
            bag_xx, bag_ll = self._bin_descriptors(siftgeo, pca, grid, (W, H), (sample_id.bf, sample_id.ef))
            all_x = range(1, grid[0] + 1)
            all_y = range(1, grid[1] + 1)
            all_t = range(1, grid[2] + 1)
            for ii, bin in enumerate(product(all_x, all_y, all_t)):
                if not self.exist_statistics:
                    fn = os.path.join(self.temp_path, self.bare_fn % (sample, grid[0], grid[1], grid[2], ii))
                    try:
                        with open(fn) as ff:
                            pass
                    except IOError:
                        with open(fn, 'w') as ff:
                            try:
                                ss = self.model._compute_statistics(
                                    vstack(bag_xx[bin]), gmm)
                            except ValueError:
                                # The current window cell contains no descriptors.
                                #print 'ValueError %s' % fn
                                ss = np.array(zeros(self.K + 2 * self.nr_pca_comps * self.K), dtype=np.float32)
                            ss.tofile(ff)
                if self.model.is_spatial_model and not self.exist_spatial_statistics:
                    # Compute spatial descriptors.
                    fn = os.path.join(self.temp_path, self.spatial_bare_fn % (sample, grid[0], grid[1], grid[2], ii))
                    try:
                        with open(fn) as ff:
                            pass
                    except IOError:
                        with open(fn, 'w') as ff:
                            try:
                                ss = self.model._compute_spatial_statistics(vstack(bag_xx[bin]), vstack(bag_ll[bin]), gmm)
                            except ValueError:
                                # The current window cell contains no descriptors.
                                #print 'ValueError %s' % fn
                                ss = np.array(zeros(self.K + 2 * 3 * self.K), dtype=np.float32)
                            ss.tofile(ff)

    def _bin_descriptors(self, siftgeo, pca, grid, dimensions, duration):
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
        bag_xx = defaultdict(list)
        bag_ll = defaultdict(list)
        for ss in siftgeo:
            xx = pca.transform(ss[1])
            id_x = digitize([ss[0]['x']], bins_x)
            id_y = digitize([ss[0]['y']], bins_y)
            id_t = digitize([ss[0]['t']], bins_t)
            bag_xx[(id_x[0], id_y[0], id_t[0])].append(xx)
            bag_ll[(id_x[0], id_y[0], id_t[0])].append([ss[0]['x'] / W, ss[0]['y'] / H, (ss[0]['t'] - t_init) / (t_final + 1 - t_init)])
            assert (1 <= id_x <= grid[0] and
                    1 <= id_y <= grid[1] and
                    1 <= id_t <= grid[2])
        return bag_xx, bag_ll

    def _merge_tmp_statistics(self, data_type, prefix=''):
        """ Data type can be either 'train' or 'test'. """
        nr_bins = prod(self.grid)
        # File format is <name>_<train/test>_<g0>_<g1>_<g2>_<ii>.

        # Clamp the first 5 parameters. Let free only the index of bin number.
        fn_first = '_'.join(self.bare_fn.split('_')[:-1]) 
        fn_last = self.bare_fn.split('_')[-1]
        name = (fn_first % (prefix + data_type, self.grid[0], self.grid[1], self.grid[2]) + '_' + fn_last)
        fn_stats = os.path.join(self.root_path, name) # Filename of big file.

        # Iterate over cells/bins combinations.
        for ii in xrange(nr_bins):
            try:
                ff = open(fn_stats % ii);
                print 'File %s already exists. Chickening out.' % fn_stats % ii
                ff.close()
                return False
            except IOError:
                pass
            ff_stats = open(fn_stats % ii, 'w') # Big file containing statistics. 
            for sample in self.dataset.get_data(data_type)[0]:
                fn = os.path.join(
                    self.temp_path, prefix + self.bare_fn % (sample, self.grid[0], self.grid[1], self.grid[2], ii))  # Small file filename
                try:
                    with open(fn) as ff:
                        ss = np.fromfile(ff, dtype=np.float32)
                        # Check everything is fine as size.
                        if prefix == '':
                            assert len(ss) == self.K + 2 * self.K * self.nr_pca_comps
                        elif prefix == 'spatial_':
                            assert len(ss) == self.K + 2 * self.K * 3
                        else:
                            print 'Unrecognized prefix.\n Exiting...'
                            sys.exit(1)
                        # Write to the big file.
                        ss.tofile(ff_stats)
                except IOError:
                    print 'File %s does not exist.' % fn
                    print 'Removing big file containing statistics %s' % fn_stats % ii
                    ff_stats.close()
                    os.remove(fn_stats % ii)
                    print 'Exiting...'
                    sys.exit(1)
                except AssertionError:
                    print 'Size mistmatch for file %s' % fn
                    print 'Removing big file containing statistics %s' % fn_stats % ii
                    ff_stats.close()
                    os.remove(fn_stats % ii)
                    print 'Exiting...'
                    sys.exit(1)
            ff_stats.close()

    def _remove_tmp_statistics(self, prefix=''):
        samples = list(set(
            self.dataset.get_data('train')[0] +
            self.dataset.get_data('test')[0]))
        nr_bins = prod(self.grid)
        for sample in samples:
            for ii in xrange(nr_bins):
                fn = os.path.join(
                    self.temp_path, prefix + self.bare_fn %
                    (sample, self.grid[0], self.grid[1], self.grid[2], ii))
                try:
                    os.remove(fn)
                except OSError:
                    pass

    def _exist_descriptors(self):
        samples = (self.dataset.get_data('train')[0] +
                   self.dataset.get_data('test')[0])
        for sample in samples:
            try:
                with open(os.path.join(self.dataset.FEAT_DIR, 'features',
                                       sample + '.siftgeo'), 'r'):
                    pass
            except IOError:
                return False
        return True

    def _exist_statistics(self, prefix=''):
        sw = True
        nr_bins = prod(self.grid)
        for ii in xrange(nr_bins):
            for data_type in ['train', 'test']:
                try:
                    ff = open(
                        os.path.join(self.root_path, prefix + self.bare_fn % (
                            data_type, self.grid[0], self.grid[1], 
                            self.grid[2], ii)))
                    ff.close()
                except:
                    sw = False
        return sw

    def _compute_subsample_descriptors(self, nr_samples):
        """ Gets a subsample of the the descriptors and it is saved to the
        subset.siftgeo file. """
        bash_cmd = ('/home/clear/oneata/scripts/bash_functions/' +
                    'run_subsample.sh %s %s %s' %
                    (self.dataset.DATASET, self.dataset.FTYPE,
                     str(nr_samples)))
        os.system(bash_cmd)

    def _load_subsample_descriptors(self):
        """ Returns a NxD dimensional matrix representing a subsample of the
        descriptors. """
        with open(self.fn_subsample, 'r'):
            pass
        siftgeos = read_video_points_from_siftgeo(self.fn_subsample)
        N = len(siftgeos)
        D = len(siftgeos[0][1])
        descriptors = zeros((N, D), dtype=np.float32)
        for ii, siftgeo in enumerate(siftgeos):
            descriptors[ii] = siftgeo[1]
        return descriptors
