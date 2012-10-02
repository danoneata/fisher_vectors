#!/usr/bin/python
""" Functions that deal with the GMM computation for a set of descriptors. """
import multiprocessing
import getopt
import os
import sys

from dataset import Dataset
from subset import load_subsample_descriptors
from pca import load_pca
from yael.yael import numpy_to_fvec_ref, gmm_learn, gmm_read, gmm_write
from yael.yael import GMM_FLAGS_W


def compute_gmm(data, nr_clusters, nr_iterations, nr_threads, seed, nr_redos):
    """ Computes GMM using yael functions. """
    N, D = data.shape
    gmm = gmm_learn(D, N, nr_clusters, nr_iterations, numpy_to_fvec_ref(data),
                    nr_threads, seed, nr_redos, GMM_FLAGS_W)
    return gmm


def save_gmm(gmm, filename):
    """ Saves the GMM object to file. """
    with open(filename, 'w') as _file:
        gmm_write(gmm, _file)


def load_gmm(filename):
    """ Loads GMM object from file using yael. """
    with open(filename, 'r') as _file:
        gmm = gmm_read(_file)
        return gmm


def usage():
    prog_name = os.path.basename(sys.argv[0])
    print "Usage: %s --dataset=SRC_CFG --nr_clusters=K" % prog_name
    print
    print "Computes GMM on a subset of the descriptors for the specified"
    print "dataset. Note: it implicitly applies PCA to the data to reduce the"
    print "dimensionality to 64 dimensions."
    print
    print "Oprtions:"
    print "     -d, --dataset=SRC_CFG"
    print "         Configuration file for the dataset."
    print
    print "     -i, --ip_type=FEAT_TYPE"
    print "         Type of features. Default, dense5.track15mbh."
    print
    print "     -k, --nr_clusters=K"
    print "         Number of clusters in the GMM."
    print
    print "     -n, --nr_pca_components=N"
    print "         Number of PCA components, default 64."
    print
    print "     --suffix=SUFFIX"
    print "         Suffix that is added to the default feature directory."
    print "         Default, no suffix."


def compute_gmm_given_dataset(src_cfg, nr_clusters, **kwargs):
    """ Uses the conventions from the dataset for loading a subset of
    descriptors, loading the PCA object and choosing an output file.

    """
    # Set default parameters.
    ip_type = kwargs.get('ip_type', 'dense5.track15mbh')
    nr_iterations = kwargs.get('nr_iterations', 100)
    nr_threads = kwargs.get('nr_threads', multiprocessing.cpu_count())
    seed = kwargs.get('seed', 1)
    nr_redos = kwargs.get('nr_redos', 4)
    suffix = kwargs.get('suffix', '')
    nr_pca_components = kwargs.get('nr_pca_components', 64)

    dataset = Dataset(src_cfg, ip_type=ip_type, suffix=suffix)
    filename_pca = os.path.join(dataset.FEAT_DIR, 'pca', 'pca_%d.pkl' % nr_pca_components)

    data = load_subsample_descriptors(dataset)
    pca = load_pca(filename_pca)
    transformed_data = pca.transform(data)

    # Do the computation.
    gmm = compute_gmm(
        transformed_data, nr_clusters, nr_iterations,
        nr_threads, seed, nr_redos)

    outfilename = os.path.join(
        dataset.FEAT_DIR, 'gmm', 'gmm_%d' % nr_clusters)
    save_gmm(gmm, outfilename)


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], 'hd:i:k:n:',
            ['help', 'dataset=', 'ip_type=', 'nr_clusters=', 'suffix='])
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
        elif opt in ('-k', '--nr_clusters'):
            nr_clusters = int(arg)
        elif opt in ('-n', '--nr_pca_components'):
            kwargs['nr_pca_components'] = int(arg)
        elif opt in ('--suffix'):
            kwargs['suffix'] = arg

    compute_gmm_given_dataset(src_cfg, nr_clusters, **kwargs)


if __name__ == '__main__':
    main()
