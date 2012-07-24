#!/usr/bin/python
""" Functions that deal with the PCA computation for a set of descriptors. """
import cPickle as pickle
import getopt
import os
import sys

from sklearn.decomposition import PCA

from dataset import Dataset
from subset import load_subsample_descriptors


def compute_pca(data, n_components):
    """ Computes PCA on a subset of nr_samples of descriptors. """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


def save_pca(pca, filename):
    """ Pickles a PCA object to the specified filename. """
    with open(filename, 'w') as ff:
        pickle.dump(pca, ff)


def load_pca(filename):
    """ Loads PCA object from file using cPickle. """
    with open(filename, 'r') as ff:
        pca = pickle.load(ff)
        return pca


def usage():
    prog_name = os.path.basename(sys.argv[0])
    print "Usage: %s --dataset=SRC_CFG" % prog_name
    print
    print "Computes PCA on a subset of the descriptors for the specified"
    print "dataset."
    print
    print "Oprtions:"
    print "     -d, --dataset=SRC_CFG"
    print "         Configuration file for the dataset."
    print
    print "     -i, --ip_type=FEAT_TYPE"
    print "         Type of features. Default, dense5.track15mbh."
    print
    print "     -n, --n_components=NR_PCA_COMP"
    print "         Number of principal components. Default, 64."
    print
    print "     --suffix=SUFFIX"
    print "         Suffix that is added to the default feature directory."
    print "         Default, no suffix."

def compute_pca_given_dataset(src_cfg, **kwargs):
    """ Uses the conventions from the dataset for loading a subset of
    descriptors and choosing an output file.

    """
    # Set default parameters.
    ip_type = kwargs.get('ip_type', 'dense5.track15mbh')
    n_components = kwargs.get('n_components', 64)
    suffix = kwargs.get('suffix', '')

    dataset = Dataset(src_cfg, ip_type=ip_type, suffix=suffix)
    data = load_subsample_descriptors(dataset)

    # Do the computation.
    pca = compute_pca(data, n_components)

    outfilename = os.path.join(
        dataset.FEAT_DIR, 'pca', 'pca_%d.pkl' % n_components)
    save_pca(pca, outfilename)


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], 'hd:i:n:',
            ['help', 'dataset=', 'ip_type=', 'n_components=', 'suffix='])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    kwargs = {}
    for opt, arg in opt_pairs:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-d', '--dataset'):
            src_cfg = arg
        elif opt in ('-i', '--ip_type'):
            kwargs['ip_type'] = arg
        elif opt in ('-n', '--n_components'):
            kwargs['n_components'] = int(arg)
        elif opt in ('--suffix'):
            kwargs['suffix'] = arg

    compute_pca_given_dataset(src_cfg, **kwargs)


if __name__ == '__main__':
    main()
