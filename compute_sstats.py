#!/usr/bin/python
from __future__ import division

import getopt
import multiprocessing
import os
import re
import subprocess
import sys

import numpy as np
from numpy import digitize
from numpy import linspace
from numpy.testing import assert_allclose
from ipdb import set_trace

from model import Model
from dataset import Dataset
from data import SstatsMap

from preprocess.pca import load_pca
from preprocess.gmm import load_gmm
from preprocess.subset import DESCS_LEN

from utils.profile import profile

from vidbase.vidplayer import get_video_infos


def parse_ip_type(ip_type):
    """ Splits an ip_type string into arguments that are passable to Heng's
    densetracks code. We follow the convention defined by Adrien. Some examples
    of ip_type: 'dense5.track15hoghof', 'dense5.track20mbh'.

    Note: At the moment, I assume we are using only dense trajectories.

    """
    try:
        detector, descriptor = ip_type.split('.')
    except ValueError:
        print 'Incorect format of ip_type.'
        sys.exit()

    assert detector.startswith('dense') and descriptor.startswith('track'), \
            'Accepts only dense trajectories at the moment.'

    pattern_stride = re.compile('dense(\d+)')
    pattern_track_length = re.compile('track(\d+)\w+')
    pattern_desc_type = re.compile('track\d+(\w+)')

    stride = re.findall(pattern_stride, detector)[0]
    track_length = re.findall(pattern_track_length, descriptor)[0]
    descriptor_type = re.findall(pattern_desc_type, descriptor)[0]

    return stride, track_length, descriptor_type


def read_descriptors_from_video(infile, **kwargs):
    """ Lazy function generator to grab chunks of descriptors from Heng's dense
    trajectories stdout. The code assumes that 'densetracks' outputs 3 numbers
    corresponding to the descriptor position, followed by the descriptor's
    values. Each outputed number is assumed to be a float.

    Parameters
    ----------
    infile: string, required
        The path to the video file.

    nr_descriptors: int, optional, default 1000
        Number of descriptors to be returned.

    ip_type: string, optional, default 'dense5.track15mbh'
        The type of descriptors to be returned.

    begin_frames: list, optional, default [0]
        The indices of the beginning frames.

    end_frames: list, optional, default [1e6]
        The indices of the end frames.

    nr_skip_frames: int, optional, default 0
        The number of frames that are skipped; for every (nr_skip_frames + 1)
        frames, (nr_skip_frames) are ignored.

    """
    FLOAT_SIZE = 4

    # Get keyword arguments or set default values.
    nr_descriptors = kwargs.get('nr_descriptors', 1000)
    ip_type = kwargs.get('ip_type', 'dense5.track15mbh')
    begin_frames = kwargs.get('begin_frames', [0])
    end_frames = kwargs.get('end_frames', [1e6])
    nr_skip_frames = kwargs.get('nr_skip_frames', 0)

    # Prepare arguments for Heng's code.
    stride, track_length, descriptor_type = parse_ip_type(ip_type)
    str_begin_frames = '_'.join(map(str, begin_frames))
    str_end_frames = '_'.join(map(str, end_frames))

    descriptor_length = DESCS_LEN[descriptor_type]
    position_length = 3

    dense_tracks = subprocess.Popen(
        ['densetracks', infile, '0', track_length, stride,
         str_begin_frames, str_end_frames, descriptor_type, '1',
         str(nr_skip_frames)],
        stdout=subprocess.PIPE, bufsize=1)
    while True:
        data = dense_tracks.stdout.read(
            FLOAT_SIZE * (descriptor_length + position_length) *
            nr_descriptors)
        if not data:
            break
        formated_data = np.fromstring(data, dtype=np.float32).reshape(
            (-1, descriptor_length + position_length))
        yield formated_data


def get_time_intervals(start, end, delta, spacing):
    """ Returns the begining and end frames for chunking the video into
    pieces of delta frames that are equally spaced.

    """
    if spacing <= 0 or delta >= end - start:
        begin_frames = [start]
        end_frames = [end]
    else:
        begin_frames = np.array(range(start, end, delta * spacing))
        end_frames = np.array(range(delta, end + delta, delta * spacing))
        end_frames = np.minimum(end_frames, end)
    return begin_frames, end_frames


def get_slice_number(current_frame, begin_frames, end_frames):
    """ Returns the index that corresponds to the slice that the current frame
    falls in.

    """
    for ii, (begin_frame, end_frame) in enumerate(zip(begin_frames,
                                                      end_frames)):
        if begin_frame <= current_frame < end_frame:
            return ii
    # return -1
    # Why did you do this, past Dan? Maybe future Dan will explain.
    # I still have no clue today, on the 6th of June.
    raise Exception('Frame number not in the specified intervals.')


def get_sample_label(dataset, sample):
    tr_samples, tr_labels = dataset.get_data('train')
    te_samples, te_labels = dataset.get_data('test')

    samples = tr_samples + te_samples
    labels = tr_labels + te_labels

    label = ()
    for _sample, _label in zip(samples, labels):
        if str(sample) == str(_sample):
            label += (_label, )

    if label is ():
        raise Exception('Sample was not found in the dataset.')
    return label


@profile
def compute_statistics(src_cfg, **kwargs):
    """ Computes sufficient statistics needed for the bag-of-words or
    Fisher vector model.

    """
    # Default parameters.
    ip_type = kwargs.get('ip_type', 'dense5.track15mbh')
    nr_clusters = kwargs.get('nr_clusters', 128)

    dataset = Dataset(src_cfg, ip_type=ip_type)
    dataset.VOC_SIZE = nr_clusters

    model_type = kwargs.get('model_type', 'fv')
    worker_type = kwargs.get('worker_type', 'normal')

    if worker_type == 'normal':
        worker = compute_statistics_from_video_worker
    elif worker_type == 'per_slice':
        from per_slice.compute_sstats_per_slice import compute_statistics_worker
        worker = compute_statistics_worker

    fn_pca = os.path.join(dataset.FEAT_DIR, 'pca', 'pca_64.pkl')
    pca = kwargs.get('pca', load_pca(fn_pca))

    fn_gmm = os.path.join(dataset.FEAT_DIR, 'gmm', 'gmm_%d' % nr_clusters)
    gmm = kwargs.get('gmm', load_gmm(fn_gmm))
    descs_to_sstats = Model(model_type, gmm).descs_to_sstats

    nr_processes = kwargs.get('nr_processes', multiprocessing.cpu_count())

    outfilename = kwargs.get('outfilename', 'stats.tmp')

    train_samples = dataset.get_data('train')[0]
    test_samples = dataset.get_data('test')[0]
    samples = list(set(train_samples + test_samples))

    sstats_out = SstatsMap(
        os.path.join(
            dataset.FEAT_DIR, 'statistics_k_%d' % nr_clusters, outfilename))

    if nr_processes > 1:
        import multiprocessing as mp
        processes = []
        nr_samples_per_process = len(samples) // nr_processes + 1
        for ii in xrange(nr_processes):
            process = mp.Process(
                target=worker,
                args=(
                    dataset,
                    samples[ii * nr_samples_per_process:
                            (ii + 1) * nr_samples_per_process],
                    sstats_out, descs_to_sstats, pca, gmm),
                kwargs=kwargs)
            processes.append(process)
            process.start()
        # Wait for jobs to finish.
        for process in processes:
            process.join()
    else:
        # We use this special case, because it makes possible to debug.
        worker(dataset, samples, sstats_out,
               descs_to_sstats, pca, gmm, **kwargs)


def compute_statistics_from_video_worker(dataset, samples, sstats_out,
                                         descs_to_sstats, pca, gmm,
                                         **kwargs):
    """ Computes the Fisher vector directly from the video in an online
    fashion. The chain of actions is the following: compute descriptors one
    by one, get a descriptor and apply PCA to it, then compute the
    posterior probabilities and update the Fisher vector.

    Inputs
    ------
    dataset: Dataset instance
        The dataset on which we are operating.

    samples: list of SampID objects
        For which samples we compute sufficietn statistics.

    sstats_out: SstatsMap instace
        Defines the output location and names.

    descs_to_sstats: callable
        Function that converts the data to sufficient statistics.

    pca: PCA instance
        Used for dimensionality reduction.

    gmm: GMM instance

    Note: it doesn't have implemented multiple grids (spatial pyramids)

    """
    nr_frames_to_skip = kwargs.get('nr_frames_to_skip', 0)
    delta = kwargs.get('delta', 0)
    spacing = kwargs.get('spacing', 0)

    D = gmm.d
    K = dataset.VOC_SIZE

    for sample in samples:
        label = get_sample_label(dataset, sample)
        # The path to the movie.
        infile = os.path.join(dataset.SRC_DIR, sample.movie + dataset.SRC_EXT)

        # Still not very nice. Maybe I should create the file on the else
        # branch.
        if sstats_out.exists(str(sample)):
            continue
        sstats_out.touch(str(sample))

        begin_frames, end_frames = get_time_intervals(
            sample.bf, sample.ef, delta, spacing)

        N = 0  # Count the number of descriptors for this sample.
        sstats = np.zeros(K + 2 * K * D, dtype=np.float32)

        for chunk in read_descriptors_from_video(
            infile, begin_frames=begin_frames, end_frames=end_frames,
            nr_skip_frames=nr_frames_to_skip):

            chunk_size = chunk.shape[0]
            # Apply PCA to the descriptor.
            xx = pca.transform(chunk[:, 3:])

            # Update the sufficient statistics for this sample.
            sstats += descs_to_sstats(xx, gmm) * chunk_size
            N += chunk_size

        sstats /= N  # Normalize statistics.
        sstats_out.write(str(sample), sstats,
                         info={'label': label, 'nr_descs': N})


def usage():
    prog_name = os.path.basename(sys.argv[0])
    print 'Usage: %s -d dataset -m model -k nr_clusters' % prog_name
    print
    print 'Computes and save sufficient statistics for a specified dataset.'
    print
    print 'Options:'
    print '     -d, --dataset=SRC_CFG'
    print '         Specify the configuration of the dataset to be loaded'
    print '         (e.g., "hollywood2_clean").'
    print
    print '     -i, --ip_type=IP_TYPE'
    print '         Specify the type of descriptor (e.g., "harris.hoghof").'
    print
    print '     -m, --model=MODEL_TYPE'
    print '         Specify the type of the model. There are the following'
    print '         possibilities:'
    print '             - "bow": bag-of-words'
    print '             - "fv": Fisher vectors'
    print '             - "bow_sfv": combination of bag-of-words and spatial'
    print '                Fisher vectors'
    print '             - "fv_sfv": combination of Fisher vectors and spatial'
    print '                Fisher vectors.'
    print '         Default, "fv."'
    print
    print '     -k, --nr_clusters=K'
    print '         Specify the number of clusters used for the dictionary.'
    print '         Default, 128.'
    print
    print '     --nr_processes=NR_PROCESSES'
    print '         Number of cores to run the operations on. By default, this'
    print '         is set to the number of nodes on the cluster.'
    print
    print "     --delta=DELTA"
    print "         Number of frames of a chunk (e.g., 120). Default 0."
    print
    print "     --spacing=SPACING"
    print "         Default 0."
    print
    print "     --nr_frames_to_skip=NR_FRAMES_TO_SKIP"
    print "         When computing descriptors, pick only one frame out of"
    print "         each (NR_FRAMES_TO_SKIP + 1) frames. Default 0."
    print
    print '     -w, --worker={"normal", "per_slice"}'
    print '         The name of the worker that performs the sufficient'
    print '         statistics computation. The "per slice" worker compute'
    print '         multiple sufficient statistics per video.'
    print
    # TODO
    print '     -g --grids=GRID'
    print '         Specify the type of spatial pyramids used. The argument'
    print '         accepts multiple grids and should be given in the'
    print '         following format: nx1_ny1_nt1[-nx2_ny2_nt2-...], where'
    print '         "n[xyt]" denotes the number of cells that are used for the'
    print '         corresponding dimension (horizontal, vertical, or'
    print '         temporal). By default, there is no spatial pyramids used'
    print '         (i.e., we use "1_1_1").'
    print '         Note: multiple grids are considered only for evaluation'
    print '         task; for the other tasks (computation, merging or'
    print '         removing) only the first grid is considered.'
    print
    # TODO
    print '     --profile'
    print '         Profiles the code using cProfile. If the number of CPUs is'
    print '         set larger than 1, the profiling is done only at a'
    print '         superficial level.'
    print
    print '     Examples:'
    print '     ---------'
    # TODO
    print '         ./compute_sstats.py -d hollywood2_clean'


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], "hd:i:m:k:o:w:",
            ["help", "dataset=", "ip_type=", "model=",
             "nr_clusters=", "nr_processes=", "delta=",
             "spacing=", "nr_frames_to_skip=", "out_filename=", "worker="])
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
        elif opt in ("-i", "--ip_type"):
            kwargs['ip_type'] = arg
        elif opt in ("-m", "--model"):
            kwargs['model_type'] = arg
        elif opt in ("-k", "--nr_clusters"):
            kwargs['nr_clusters'] = int(arg)
        elif opt in ("--nr_processes"):
            kwargs['nr_processes'] = int(arg)
        elif opt in ('--delta'):
            kwargs['delta'] = int(arg)
        elif opt in ('--spacing'):
            kwargs['spacing'] = int(arg)
        elif opt in ("--nr_frames_to_skip"):
            kwargs['nr_frames_to_skip'] = int(arg)
        elif opt in ("-o", "--out_filename"):
            kwargs['outfilename'] = arg
        elif opt in ("w", "--worker"):
            kwargs['worker_type'] = arg

    compute_statistics(src_cfg, **kwargs)


if __name__ == '__main__':
    main()
