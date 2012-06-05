#!/usr/bin/python
from __future__ import division
from collections import defaultdict
from itertools import product
import multiprocessing
import os
import re
import subprocess
import sys

import numpy as np
from numpy import digitize, linspace
from numpy import prod, vstack, zeros
from ipdb import set_trace

from model import Model
from dataset import SampID, Dataset
from data import SstatsMap
from preprocess.pca import load_pca
from preprocess.gmm import load_gmm
from vidbase.vidplayer import get_video_infos
from video_vwgeo import read_video_points_from_siftgeo
from yael.yael import count_cpu

verbose = True  # Global variable used for printing messages.

# Defining some constants for better readability.
FLOAT_SIZE = 4
DESCS_LEN = {
    'mbh': 192,
    'hog': 96,
    'hof': 108,
    'hoghof': 96 + 108,
    'all': 96 + 108 + 192}


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
            FLOAT_SIZE * (descriptor_length + position_length) * nr_descriptors)
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
        begin_frames = range(start, end - delta, delta * spacing)
        end_frames = range(start + delta, end, delta * spacing)
    return begin_frames, end_frames


def get_slice_number(current_frame, begin_frames, end_frames):
    """ Returns the index that corresponds to the slice that the current frame
    falls in.

    """
    for ii, (begin_frame, end_frame) in enumerate(zip(begin_frames,
                                                      end_frames)):
        if begin_frame <= current_frame <= end_frame:
            return ii
    #return -1   # Why did you do this, past Dan? Maybe future Dan will explain.
    raise Exception('Frame number not in the specified intervals.')


def get_sample_label(dataset, sample):
    pass


def compute_statistics(src_cfg, K, **kwargs):
    """ Computes sufficient statistics needed for the bag-of-words or
    Fisher vector model.

    """
    # Default parameters.
    ip_type = kwargs.get('ip_type', 'dense5.track15mbh')

    dataset = Dataset(src_cfg, ip_type=ip_type)
    dataset.VOC_SIZE = K

    # TODO Correct Fisher vector model.
    descs_to_sstats = kwargs.get('descs_to_sstats', Model('fv', K)._compute_statistics)
    worker = kwargs.get('worker', compute_statistics_from_video_worker)

    fn_pca = os.path.join(dataset.FEAT_DIR, 'pca', 'pca_64.pkl')
    pca = kwargs.get('pca', load_pca(fn_pca))

    fn_gmm = os.path.join(dataset.FEAT_DIR, 'gmm', 'gmm_%d' % K)
    gmm = kwargs.get('gmm', load_gmm(fn_gmm))

    grids = kwargs.get('grids', [(1, 1, 1)])
    nr_processes = kwargs.get('nr_processes', multiprocessing.cpu_count())

    train_samples = dataset.get_data('train')[0]
    test_samples = dataset.get_data('test')[0]
    samples = list(set(train_samples + test_samples))
    
    sstats_out = SstatsMap(
        os.path.join(
            dataset.FEAT_DIR, 'statistics_k_%d' % K, 'my_stats'))
    set_trace()

    # Insert here a for grid in model.grids: 
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
        infile = os.path.join( dataset.SRC_DIR, sample.movie + dataset.SRC_EXT)

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
        sstats_out.write(sstats, info={
            'label': label,  # Write a function that finds the labels for a given sample and that also accepts multilabled problems
            'nr_descs': N})


def main():
    # Parse argumets.
    pass
    compute_statistics('hollywood2_medium', 100, nr_processes=1)


if __name__ == '__main__':
    main()
