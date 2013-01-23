#!/usr/bin/python
""" Creates new sufficient statistics by averaging together sufficient
statistics from N slices.

"""
import getopt
from ipdb import set_trace
import multiprocessing as mp
import numpy as np
import os
import sys

from dataset import Dataset
from fisher_vectors.data import SstatsMap
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize
from fisher_vectors.model import FVModel
from fisher_vectors.preprocess.gmm import load_gmm


def merge(filenames, sstats_in, sstats_out, N, len_sstats, **kwargs):
    """ Aggregate together every N slices.

    Inputs
    ------
    filenames: list of str
        The names of the files to be aggregated. Usually, this should be the
        entire dataset, i.e. dataset.get_data('train')[0] +
        dataset.get_data('test')[0].

    sstats_in: SstatsMap instance
        The sufficient statistics that we operate on.

    sstats_out: SstatsMap instance
        The resulting sufficient statistics.

    N: int
        The number of slices that are aggregated together. If N is -1 all the
        slices in the clip are aggregated together.

    len_sstats: int
        The length of a typical sufficient statistics vector.

    """
    for filename in filenames:
        if sstats_out.exists(filename):
            continue

        if not sstats_in.exists(filename):
            print 'Not found ' + filename
            continue

        if sstats_in.getsize(filename) == 0:
            print 'Not computed ' + filename
            continue

        sstats = sstats_in.read(filename).reshape((-1, len_sstats))
        info = sstats_in.read_info(filename)

        agg_sstats, agg_info = _aggregate(sstats, info, N) 
        sstats_out.write(filename, agg_sstats, info=agg_info)


def double_normalization(filenames, sstats_in , sstats_out, N, len_sstats,
                         gmm):
    """ The slices in each sample are converted to Fisher vectors, square-
    rooted, L2 normalized, and then aggregated together.

    Inputs
    ------
    filenames: list of str
        The names of the files to be aggregated. Usually, this should be the
        entire dataset, i.e. dataset.get_data('train')[0] +
        dataset.get_data('test')[0].

    sstats_in: SstatsMap instance
        The sufficient statistics that we operate on.

    sstats_out: SstatsMap instance
        The resulting sufficient statistics.

    N: int
        The number of slices that are aggregated together. If N is -1 all the
        slices in the clip are aggregated together.

    len_sstats: int
        The length of a typical sufficient statistics vector.

    gmm: yael.gmm object
        The Gaussian mixture model for the current sufficient statistics.

    """
    assert len_sstats == gmm.k + 2 * gmm.k * gmm.d, (
        "GMM and len_sstats don't match")
    for filename in filenames:
        if sstats_out.exists(filename):
            continue

        if not sstats_in.exists(filename):
            print 'Not found ' + filename
            continue

        if sstats_in.getsize(filename) == 0:
            print 'Not computed ' + filename
            continue

        sstats = sstats_in.read(filename).reshape((-1, len_sstats))
        info = sstats_in.read_info(filename)

        fv = FVModel.sstats_to_features(sstats, gmm)
        fv = power_normalize(fv, 0.5)
        fv = L2_normalize(fv)

        agg_sstats, agg_info = _aggregate(fv, info, N) 
        sstats_out.write(filename, agg_sstats, info=agg_info)


def _aggregate(sstats, info, N):
    nr_slices = len(info['nr_descs'])
    nr_non_zeros_slices, nr_features = sstats.shape
    augment_sstats = np.zeros((nr_slices, nr_features))
    augment_sstats[np.where(info['nr_descs'] != 0)] = sstats

    # Number of slices after aggregation.
    if N == -1:
        N = nr_slices
    kk = int(np.ceil(float(nr_slices) / N))

    idxs = []
    for ii in xrange(kk):
        idxs += [ii] * N
    idxs = idxs[: nr_slices]

    matrix_nr_descs = np.zeros((kk, nr_slices))
    matrix_nr_descs[idxs, np.arange(nr_slices)] = info['nr_descs']

    begin_idxs = np.arange(0, nr_slices, N) 
    end_idxs = np.minimum(begin_idxs + N - 1, nr_slices - 1)

    agg_info = {}
    agg_info['nr_descs'] = np.sum(matrix_nr_descs, 1)
    agg_info['begin_frames'] = np.array(info['begin_frames'])[begin_idxs]
    agg_info['end_frames'] = np.array(info['end_frames'])[end_idxs]
    agg_info['label'] = info['label']

    agg_nr_descs = agg_info['nr_descs']
    agg_sstats = np.dot(
        matrix_nr_descs, augment_sstats)[agg_nr_descs != 0] / agg_nr_descs[
            agg_nr_descs != 0][:, np.newaxis]

    assert len(
        agg_nr_descs) == len(agg_info['begin_frames']), 'Dimensions mismatch'
    assert len(
        agg_nr_descs) == len(agg_info['end_frames']), 'Dimensions mismatch'
    assert len(set(agg_info['end_frames'] - agg_info['begin_frames'])) <= 2

    return agg_sstats, agg_info
    

def master(src_cfg, suffix_in ,suffix_out, K, N, nr_processes, double_norm):
    D = 64

    dataset = Dataset(src_cfg, nr_clusters=K)
    samples = [str(sample) for sample in dataset.get_data('train')[0] +
               dataset.get_data('test')[0]]

    if double_norm:
        worker = double_normalization
        suffix = '.double_norm'
        gmm = load_gmm(
            os.path.join(
                dataset.FEAT_DIR + suffix_in, 'gmm',
                'gmm_%d' % K))
    else:
        worker = merge
        suffix = ''
        gmm = None

    path_in = os.path.join(
        dataset.FEAT_DIR + suffix_in,
        'statistics_k_%d' % dataset.VOC_SIZE, 'stats.tmp')
    path_out = os.path.join(
        dataset.FEAT_DIR + suffix_out,
        'statistics_k_%d' % dataset.VOC_SIZE, 'stats.tmp' + suffix)

    sstats_in = SstatsMap(path_in)
    sstats_out = SstatsMap(path_out)

    len_sstats = dataset.VOC_SIZE + 2 * D * dataset.VOC_SIZE

    kwargs = {
        'N': N,
        'sstats_in': sstats_in,
        'sstats_out': sstats_out,
        'len_sstats': len_sstats,
        'gmm': gmm}

    if nr_processes > 1:
        nr_samples_per_process = len(samples) / nr_processes + 1
        for ii in xrange(nr_processes):
            mp.Process(target=worker,
                       args=(samples[
                           ii * nr_samples_per_process:
                           (ii + 1) * nr_samples_per_process], ),
                       kwargs=kwargs).start()
    else:
        worker(samples, **kwargs)


def usage():
    pass


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], "hd:n:k:",
            ["help", "dataset=", "nr_slices_to_merge=", "nr_clusters=",
             "nr_processes=", "suffix_in=", "suffix_out=", "double_norm"])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)
    
    double_norm = False
    nr_processes = mp.cpu_count()
    for opt, arg in opt_pairs:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-d", "--dataset"):
            src_cfg = arg
        elif opt in ("--suffix_in"):
            suffix_in = arg
        elif opt in ("--suffix_out"):
            suffix_out = arg
        elif opt in ("-n", "--nr_slices_to_merge"):
            nr_slices_to_merge = int(arg)
        elif opt in ("-k", "--nr_clusters"):
            nr_clusters = int(arg)
        elif opt in ("--nr_processes"):
            nr_processes = int(arg)
        elif opt in ("--double_norm"):
            double_norm = True
    master(src_cfg, suffix_in, suffix_out, nr_clusters, nr_slices_to_merge, nr_processes, double_norm)


if __name__ == '__main__':
    main()
