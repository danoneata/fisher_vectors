#!/usr/bin/python
import cPickle
from collections import defaultdict
import getopt
import os
import sys
import numpy as np

from dataset import Dataset
from constants import NR_PCA_COMPONENTS
from constants import IP_TYPE


class SstatsMap(object):
    """ Wrapper for the temporary data files, the sufficient statistics. """
    def __init__(self, basepath):
        """ Constructor for Data object.

        Input
        -----
        basepath: str
            Path where the data will be stored.

        """
        self.basepath = basepath

        # Extensions of the files.
        self.data_ext = '.dat'
        self.info_ext = '.info'

        # If folder does not exist create it.
        if not os.path.exists(basepath):
            os.makedirs(basepath)

    def write(self, filename, data, **kwargs):
        """ Writes the data chunk and additional information to file.

        Input
        -----
        filename: str
            Name of the file where data will be stored.

        data: array [nr_sstats, nr_dimensions]
            The sufficient statistics for one sample.

        info: dict, optional
            Additional information that is stored for each sample: label,
            nr_slices, begin_frames, end_frames, nr_descs. Default, None.

        Note
        ----
        - At the moment, writing to an existing file overwrites the content.

        """
        info = kwargs.get('info', None)

        complete_path = os.path.join(self.basepath, filename)
        data_file = open(complete_path + self.data_ext, 'w')
        np.array(data, dtype=np.float32).tofile(data_file)
        data_file.close()

        if info is not None:
            info_file = open(complete_path + self.info_ext, 'w')
            cPickle.dump(info, info_file)
            info_file.close()

    def write_info(self, filename, info):
        """ Writes only information data. """
        complete_path = os.path.join(self.basepath, filename + self.info_ext)
        ff = open(complete_path, 'w')
        cPickle.dump(info, ff)
        ff.close()

    def touch(self, filename):
        """ Creates a file with the given filename. """
        data_file = os.path.join(self.basepath, filename + self.data_ext)
        os.system('touch %s' % data_file)

    def exists(self, filename):
        """ Checks whether the file exists or not. """
        data_file = os.path.join(self.basepath, filename + self.data_ext)
        return os.path.exists(data_file)

    def read(self, filename):
        """ Reads data from file. """
        complete_path = os.path.join(self.basepath, filename)
        data_file = open(complete_path + self.data_ext, 'r')
        data = np.fromfile(data_file, dtype=np.float32)
        data_file.close()
        return data

    def read_info(self, filename):
        """ Reads additional information from the file. """
        complete_path = os.path.join(self.basepath, filename)
        info_file = open(complete_path + self.info_ext, 'r')
        info = cPickle.load(info_file)
        info_file.close()
        return info

    def check(self, filenames, len_sstats, **kwargs):
        """ Performs simple checks of the data for the given filenames.

        Inputs
        ------
        filenames: str
            Files to be checked.

        len_sstats: int
            Length of sufficient statistics.

        print_incorrect: boolean, optional
            Prints incorrectly computed sufficient statistics.

        print_missing: boolean, optional
            Prints missing sufficient statistics.

        Output
        ------
        status: boolean
            Returns True if all the files pass the verification and False
            otherwise.

        """
        print_missing = kwargs.get('print_missing', True)
        print_incorrect = kwargs.get('print_incorrect', True)

        status = True
        missing_files = []
        incorrect_files = []

        for filename in filenames:
            try:
                data = self.read(filename)
                nr_elems = len(data)
                if (nr_elems == 0 or
                    nr_elems % len_sstats != 0 or
                    np.isnan(np.max(data))):
                    status = False
                    incorrect_files.append(filename + self.data_ext)
            except IOError:
                status = False
                missing_files.append(filename + self.data_ext)

        if print_incorrect:
            print 'Incorrect values in the following files:'
            print ' '.join(incorrect_files)

        if print_missing:
            print 'Missing files:'
            print ' '.join(missing_files)

        return status

    def get_merged(self, filenames, len_sstats, **kwargs):
        """ Returns list for the merged sufficient statisitcs, labels, video
        limits, video names, number of descriptors, begin frames and end
        frames. 

        Note 1: Slices that have no descriptors are ignored.

        Note 2: Returned labels are not tuples.
        
        """
        aggregate = kwargs.get('aggregate', False)
        # TODO Mantain consistency with merge function.
        all_sstats = []
        all_labels = []
        outinfo = defaultdict(list)
        outinfo['limits'] = [0]
        sstats = []
        for filename in filenames:
            sstats = self.read(filename)
            info = self.read_info(filename)

            nr_descs = info['nr_descs']
            label = info['label']
            outinfo['video_names'].append(filename.split('-')[0])

            if aggregate:
                sstats = self._aggregate_sstats(sstats, nr_descs, len_sstats)
                outinfo['nr_descs'].append(np.sum(nr_descs))
                outinfo['begin_frames'].append(info['begin_frames'][0])
                outinfo['end_frames'].append(info['end_frames'][-1])
            else:
                outinfo['nr_descs'].append(nr_descs[nr_descs != 0])
                outinfo['begin_frames'].append(
                    np.array(info['begin_frames'])[nr_descs != 0])
                outinfo['end_frames'].append(
                    np.array(info['end_frames'])[nr_descs != 0])

            nr_elems = len(sstats)
            nr_of_slices = nr_elems / len_sstats
            assert nr_elems % len_sstats == 0, ("The length of the sufficient"
                                              "statistics is not a multiple of"
                                              "the length of the descriptor.")
            if not aggregate:
                assert nr_of_slices == len(nr_descs[nr_descs != 0]), (
                    "Incorrect number of descriptors.")

            all_labels += [label for ii in xrange(nr_of_slices)]
            outinfo['limits'].append(outinfo['limits'][-1] + nr_of_slices)
            all_sstats.append(sstats.reshape(-1, len_sstats))

        if not aggregate:
            outinfo['nr_descs'] = np.hstack(outinfo['nr_descs'])
            outinfo['begin_frames'] = np.hstack(outinfo['begin_frames'])
            outinfo['end_frames'] = np.hstack(outinfo['end_frames'])

        return np.vstack(all_sstats), all_labels, outinfo

    def merge(self, filenames, outfilename, len_sstats, **kwargs):
        """ Merges a specified set of filenames.

        Inputs
        ------
        samples: list of strings
            The samples names whose sufficient statistics will be merged.

        outfilename: str
            Name of the output file.

        outfolder: str, optional
            Name of the output folder. By default, the output folder is the
            same as the folder where the statistics are saved.

        aggregate: boolean, optional
            Specifies if it should aggregate per slices statistics together.

        Note:
        1. The function also computes the associated file with labels. This
        file will be named as 'labels_' + `outfilename`.
        
        2. The output are the merged sufficient statistics, labels and other
        useful information: video name, begin frames, end frames, number of
        descriptors, limits.

        """
        outfolder = kwargs.get('outfolder', self.basepath)
        aggregate = kwargs.get('aggregate', False)

        sstats_filename = os.path.join(outfolder, outfilename + self.data_ext)
        labels_filename = os.path.join(outfolder, 'labels_'
                                       + outfilename + self.info_ext)
        others_filename = os.path.join(outfolder, 'info_'
                                       + outfilename + self.info_ext)

        # If sstats_file exists delete it.
        if os.path.exists(sstats_filename):
            os.remove(sstats_filename)

        sstats_file = open(sstats_filename, 'a')
        labels_file = open(labels_filename, 'w')
        others_file = open(others_filename, 'w')

        all_labels = []
        outinfo = defaultdict(list)
        outinfo['limits'] = [0]
        for filename in filenames:
            sstats = self.read(filename)
            info = self.read_info(filename)

            nr_descs = info['nr_descs']
            label = info['label']
            outinfo['video_names'].append(filename.split('-')[0])

            if aggregate:
                sstats = self._aggregate_sstats(sstats, nr_descs, len_sstats)
                outinfo['nr_descs'].append(np.sum(nr_descs))
                outinfo['begin_frames'].append(info['begin_frames'][0])
                outinfo['end_frames'].append(info['end_frames'][-1])
            else:
                outinfo['nr_descs'].append(nr_descs[nr_descs != 0])
                from ipdb import set_trace
                outinfo['begin_frames'].append(
                    np.array(info['begin_frames'])[nr_descs != 0])
                outinfo['end_frames'].append(
                    np.array(info['end_frames'])[nr_descs != 0])

            nr_elems = len(sstats)
            nr_of_slices = nr_elems / len_sstats
            assert nr_elems % len_sstats == 0, ("The length of the sufficient"
                                              "statistics is not a multiple of"
                                              "the length of the descriptor.")
            if not aggregate:
                assert nr_of_slices == len(nr_descs[nr_descs != 0]), (
                    "Incorrect number of descriptors.")

            all_labels += [label for ii in xrange(nr_of_slices)]
            outinfo['limits'].append(outinfo['limits'][-1] + nr_of_slices)
            sstats.tofile(sstats_file)

        if not aggregate:
            outinfo['nr_descs'] = np.hstack(outinfo['nr_descs'])
            outinfo['begin_frames'] = np.hstack(outinfo['begin_frames'])
            outinfo['end_frames'] = np.hstack(outinfo['end_frames'])

        cPickle.dump(all_labels, labels_file)
        cPickle.dump(outinfo, others_file)
        sstats_file.close()
        labels_file.close()
        others_file.close()

    def _aggregate_sstats(self, sstats, nr_descs, len_sstats):
        sstats = sstats.reshape((-1, len_sstats))
        nn = nr_descs[nr_descs != 0]
        sstats = np.array(np.sum(sstats * nn[:, np.newaxis], 0) / np.sum(nn),
                          dtype=np.float32)
        return sstats


def merge_given_dataset(dataset, **kwargs):
    """ Merges the statistics and the labels for the train and the test set.

    """
    basepath = os.path.join(dataset.SSTATS_DIR, 'stats.tmp')
    outfolder = dataset.SSTATS_DIR
    data = SstatsMap(basepath)
    nr_clusters = dataset.VOC_SIZE

    tr_samples = dataset.get_data('train')[0]
    str_tr_samples = list(set([str(sample) for sample in tr_samples]))
    data.merge(str_tr_samples, 'train', nr_clusters +
               2 * nr_clusters * NR_PCA_COMPONENTS, outfolder=outfolder)
    print "Merged train data."

    te_samples = dataset.get_data('test')[0]
    str_te_samples = list(set([str(sample) for sample in te_samples]))
    data.merge(str_te_samples, 'test', nr_clusters +
               2 * nr_clusters * NR_PCA_COMPONENTS, outfolder=outfolder)
    print "Merged test data."


def check_given_dataset(dataset, **kwargs):
    """ Checks the train and the test samples. """
    basepath = os.path.join(dataset.SSTATS_DIR, 'stats.tmp')
    data = SstatsMap(basepath)
    nr_clusters = dataset.VOC_SIZE

    tr_samples = dataset.get_data('train')[0]
    str_tr_samples = list(set([str(sample) for sample in tr_samples]))
    tr_status = data.check(str_tr_samples, nr_clusters +
                           2 * nr_clusters * NR_PCA_COMPONENTS, **kwargs)

    te_samples = dataset.get_data('test')[0]
    str_te_samples = list(set([str(sample) for sample in te_samples]))
    te_status = data.check(str_te_samples, nr_clusters +
                           2 * nr_clusters * NR_PCA_COMPONENTS, **kwargs)

    if tr_status and te_status:
        print 'Checking done. Everything ok.'


def usage():
    prog_name = os.path.basename(sys.argv[0])
    print 'Usage: %s -d dataset -k nr_clusters' % prog_name
    print
    print 'Computes and save sufficient statistics for a specified dataset.'
    print
    print 'Options:'
    print '     -d, --dataset=SRC_CFG'
    print '         Specify the configuration of the dataset to be loaded'
    print '         (e.g., "hollywood2_clean").'
    print
    print '     -k, --nr_clusters=K'
    print '         Specify the number of clusters used for the dictionary.'
    print
    print '     -t, --task={"check", "merge"}'
    print '         Perform one of the two tasks: check sufficient statistics'
    print '         or merge them.'
    print
    print '     -o, --out_folder=PATH'
    print '         Option only for "merge" task. Location where the merged'
    print '         files will be stored. Optional parameter.'
    print
    print '     --suffix=SUFFIX'
    print '         Appends a suffix to the standard feature path name.'
    print
    print '     --missing'
    print '         When checking the sufficient statistics, prints only those'
    print '         statistics that are missing.'
    print
    print '     --incorrect'
    print '         When checking the sufficient statistics, prints only those'
    print '         statistics that are incorrect, contain NaNs.'


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], "hd:k:t:o:",
            ["help", "dataset=", "nr_clusters=", "task=", "out_folder=",
             "suffix=", "missing", "incorrect"])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    kwargs = {}
    suffix = ''
    for opt, arg in opt_pairs:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-d", "--dataset"):
            src_cfg = arg
        elif opt in ("-k", "--nr_clusters"):
            nr_clusters = int(arg)
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-o", "--out_folder"):
            kwargs["outfolder"] = arg
        elif opt in ("--suffix"):
            suffix = arg
        elif opt in ("--missing"):
            kwargs["print_missing"] = True
            kwargs["print_incorrect"] = False
        elif opt in ("--incorrect"):
            kwargs["print_missing"] = False
            kwargs["print_incorrect"] = True

    if task not in ("check", "merge"):
        print "Unknown task."
        usage()
        sys.exit(1)

    dataset = Dataset(
        src_cfg, ip_type=IP_TYPE, suffix=suffix, nr_clusters=nr_clusters)
    if task == "check":
        check_given_dataset(dataset, **kwargs)
    elif task == "merge":
        merge_given_dataset(dataset, **kwargs)


if __name__ == '__main__':
    main()
