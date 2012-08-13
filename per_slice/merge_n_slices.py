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


def merge(filenames, sstats_in, sstats_out, N, len_sstats):
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
            print filename
            continue

        sstats = sstats_in.read(filename).reshape((-1, len_sstats))
        info = sstats_in.read_info(filename)

        agg_sstats, agg_info = _aggregate(sstats, info, N) 
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
    

def master(nr_processes):
    D = 64
    K = 256
    N = -1 

    dataset = Dataset('trecvid12', nr_clusters=K)
    samples = [str(sample) for sample in dataset.get_data('train')[0] +
               dataset.get_data('test')[0]]

    path_in = os.path.join(
        dataset.FEAT_DIR + '.per_slice.small.delta_60.skip_3',
        'statistics_k_%d' % dataset.VOC_SIZE, 'stats.tmp')
    path_out = os.path.join(
        dataset.FEAT_DIR + '.small.skip_3',
        'statistics_k_%d' % dataset.VOC_SIZE, 'stats.tmp')

    sstats_in = SstatsMap(path_in)
    sstats_out = SstatsMap(path_out)

    len_sstats = dataset.VOC_SIZE + 2 * D * dataset.VOC_SIZE

    if nr_processes > 1:
        nr_samples_per_process = len(samples) / nr_processes + 1
        for ii in xrange(nr_processes):
            mp.Process(target=merge,
                       args=(samples[
                           ii * nr_samples_per_process:
                           (ii + 1) * nr_samples_per_process],
                           sstats_in, sstats_out, N, len_sstats)).start()
    else:
        merge(samples, sstats_in, sstats_out, N, len_sstats)


def usage():
    pass


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], "hn:", ["help", "nr_processes="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)
    
    nr_processes = mp.cpu_count()
    for opt, arg in opt_pairs:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-n", "--nr_processes"):
            nr_processes = int(arg)
    master(nr_processes)


if __name__ == '__main__':
    main()
