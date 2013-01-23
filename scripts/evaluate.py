#!/usr/bin/python
import cPickle as pickle
import getopt
import os
import sys
from ipdb import set_trace

from numpy.testing import assert_allclose

from dataset import Dataset

from fisher_vectors.model import Model
from fisher_vectors.preprocess.gmm import load_gmm
from fisher_vectors.evaluation import Evaluation


def evaluate_given_dataset(dataset, **kwargs):
    model_type = kwargs.get('model_type', 'fv')

    sstats_folder = dataset.SSTATS_DIR

    tr_fn = os.path.join(sstats_folder, 'train.dat')
    tr_labels_fn = os.path.join(sstats_folder, 'labels_train.info')

    te_fn = os.path.join(sstats_folder, 'test.dat')
    te_labels_fn = os.path.join(sstats_folder, 'labels_test.info')

    gmm = load_gmm(dataset.GMM)

    tr_labels = pickle.load(open(tr_labels_fn, 'r'))
    te_labels = pickle.load(open(te_labels_fn, 'r'))

    model = Model(model_type, gmm)
    model.compute_kernels([tr_fn], [te_fn])
    Kxx, Kyx = model.get_kernels()

    evaluation = Evaluation(dataset.DATASET, **kwargs)
    print evaluation.fit(Kxx, tr_labels).score(Kyx, te_labels)


def usage():
    pass


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1: ], "hd:k:i:m:",
            ["help", "dataset=", "nr_clusters=",
             "ip_type=", "model="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    kwargs = {}
    for opt, arg in opt_pairs:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-d", "--dataset"):
            src_cfg = arg
        elif opt in ("-k", "--nr_clusters"):
            nr_clusters = int(arg)
        elif opt in ("-m", "--model"):
            kwargs["model_type"] = arg
        elif opt in ("-i", "--ip_type"):
            ip_type = arg
    dataset = Dataset(src_cfg, nr_clusters=nr_clusters, ip_type=ip_type)
    evaluate_given_dataset(dataset, **kwargs)


if __name__ == '__main__':
    main()
