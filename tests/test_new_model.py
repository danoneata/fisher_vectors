import cPickle as pickle
import os

from numpy.testing import assert_allclose

from dataset import Dataset

from fisher_vectors.model import Model
from fisher_vectors.preprocess.gmm import load_gmm
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.constants import IP_TYPE


class TestNewModel():
    def setup(self):
        # These scores are achieved for the following evaluation parameters.
        # class_weights='auto', C from 3 ** -2 .. 3 ** 8, and one shuffle
        # split with test size of 0.25.
        self.expected_scores = {'fv': 55.226,
                                'fv_sfv': 56.718,
                                'bow': 42.896}

        src_cfg = 'hollywood2_clean'
        nr_clusters = 100

        self.dataset = Dataset(src_cfg, ip_type='dense5.track15mbh',
                               suffix='.original', nr_clusters=nr_clusters)
        sstats_folder = self.dataset.SSTATS_DIR
        gmm_fn = self.dataset.GMM

        self.tr_fn = os.path.join(sstats_folder, 'train_1_1_1_0.dat')
        self.sp_tr_fn = os.path.join(sstats_folder,
                                     'spatial_train_1_1_1_0.dat')
        self.tr_labels_fn = os.path.join(sstats_folder, 'labels_train.info')
        self.te_fn = os.path.join(sstats_folder, 'test_1_1_1_0.dat')
        self.sp_te_fn = os.path.join(sstats_folder, 'spatial_test_1_1_1_0.dat')
        self.te_labels_fn = os.path.join(sstats_folder, 'labels_test.info')

        self.gmm = load_gmm(gmm_fn)

        tr_labels = pickle.load(open(self.tr_labels_fn, 'r'))
        te_labels = pickle.load(open(self.te_labels_fn, 'r'))

        self.cx = tr_labels
        self.cy = te_labels
        
    def test_fv(self):
        model = Model('fv', self.gmm)
        model.compute_kernels([self.tr_fn], [self.te_fn])
        Kxx, Kyx = model.get_kernels()

        evaluation = Evaluation(self.dataset.DATASET)
        score = evaluation.fit(Kxx, self.cx).score(Kyx, self.cy)
        assert_allclose(score, self.expected_scores['fv'], rtol=1e-4)

    def test_bow(self):
        model = Model('bow', self.gmm)
        model.compute_kernels([self.tr_fn], [self.te_fn])
        Kxx, Kyx = model.get_kernels()

        evaluation = Evaluation(self.dataset.DATASET)
        score = evaluation.fit(Kxx, self.cx).score(Kyx, self.cy)
        assert_allclose(score, self.expected_scores['bow'], rtol=1e-4)

    def test_fv_sfv(self):
        model = Model('fv_sfv', self.gmm)
        model.compute_kernels([self.tr_fn], [self.te_fn],
                              [self.sp_tr_fn], [self.sp_te_fn])
        Kxx, Kyx = model.get_kernels()

        evaluation = Evaluation(self.dataset.DATASET)
        score = evaluation.fit(Kxx, self.cx).score(Kyx, self.cy)
        assert_allclose(score, self.expected_scores['fv_sfv'], rtol=1e-4)
