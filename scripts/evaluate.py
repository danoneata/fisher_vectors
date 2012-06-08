#!/usr/bin/python
import cPickle as pickle
import os

from numpy.testing import assert_allclose

from dataset import Dataset

from fisher_vectors.model import Model
from fisher_vectors.preprocess.gmm import load_gmm
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.constants import IP_TYPE


def evaluate_given_dataset(src_cfg, nr_clusters, **kwargs):
    ip_type = kwargs.get('ip_type', IP_TYPE)
    dataset = Dataset(src_cfg, ip_type=ip_type)

    infolder = kwargs.get('infolder', dataset.FEAT_DIR)

    sstats_folder = os.path.join(infolder, 'statistics_k_%d' % nr_clusters)
    gmm_fn = os.path.join(infolder, 'gmm', 'gmm_%d' % nr_clusters)

    tr_fn = os.path.join(sstats_folder, 'train.dat')
    tr_labels_fn = os.path.join(labels_folder, 'labels_train.info')

    te_fn = os.path.join(sstats_folder, 'test.dat')
    te_labels_fn = os.path.join(labels_folder, 'labels_test.info')

    gmm = load_gmm(gmm_fn)

    tr_labels = pickle.load(open(tr_labels_fn, 'r'))
    te_labels = pickle.load(open(te_labels_fn, 'r'))

    model = Model('fv', gmm)
    model.compute_kernels([tr_fn], [te_fn])
    Kxx, Kyx = model.get_kernels()

    evaluation = Evaluation(dataset.DATASET)
    print evaluation.fit(Kxx, tr_labels).score(Kyx, te_labels)


def usage():
    pass


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1: ], "hd:k:i:I:",
            ["help", "dataset=", "nr_clusters=", "ip_type=", "in_folder="])
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
        elif opt in ("-i", "--ip_type"):
            kwargs["ip_type"] = arg
        elif opt in ("-I", "--in_folder"):
            kwargs["infolder"] = arg
    evaluate_given_dataset(src_cfg, nr_clusters, **kwargs)


if __name__ == '__main__':
    main()
