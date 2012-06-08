#!/usr/bin/python
import cPickle
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
            nr_slices, begin_frames, end_frames, nr_descs_per_slice. Default,
            None.

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

        verbose: boolean, optional
            Print output. Default, True.

        """
        verbose = kwargs.get('verbose', True)
        status = True
        for filename in filenames:
            data = self.read(filename)
            nr_elems = len(data)
            if (nr_elems == 0 or
                nr_elems % len_sstats != 0 or
                np.isnan(np.max(data))):
                status = False
                if verbose:
                    print filename
        return status

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

        Note: The function also computes the associated file with labels. This
        file will be named as 'labels_' + `outfilename`.

        """
        outfolder = kwargs.get('outfolder', self.basepath)

        sstats_filename = os.path.join(outfolder, outfilename + self.data_ext)
        labels_filename = os.path.join(outfolder, 'labels_'
                                       + outfilename + self.info_ext)

        # If sstats_file exists delete it.
        if os.path.exists(sstats_filename):
            os.remove(sstats_filename)

        sstats_file = open(sstats_filename, 'a')
        labels_file = open(labels_filename, 'w')

        all_labels = []
        for filename in filenames:
            sstats = self.read(filename)
            label = self.read_info(filename)['label']

            nr_elems = len(sstats)
            nr_of_slices = nr_elems / len_sstats
            assert nr_elems % len_sstats == 0, ("The length of the sufficient"
                                              "statistics is not a multiple of"
                                              "the length of the descriptor.")

            all_labels += [label for ii in xrange(nr_of_slices)]
            sstats.tofile(sstats_file)

        cPickle.dump(all_labels, labels_file)
        sstats_file.close()
        labels_file.close()


def merge_given_dataset(src_cfg, nr_clusters, **kwargs):
    """ Merges the statistics and the labels for the train and the test set.

    """
    dataset = Dataset(src_cfg, ip_type=IP_TYPE)

    basepath = os.path.join(dataset.FEAT_DIR,
                            'statistics_k_%d' % nr_clusters, 'stats.tmp')
    outfolder = os.path.join(dataset.FEAT_DIR, 'statistics_k_%d' % nr_clusters) 
    data = SstatsMap(basepath)

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


def check_given_dataset(src_cfg, nr_clusters):
    """ Checks the train and the test samples. """
    dataset = Dataset(src_cfg, ip_type=IP_TYPE)

    basepath = os.path.join(dataset.FEAT_DIR,
                            'statistics_k_%d' % nr_clusters,
                            'new_stats.old')
    data = SstatsMap(basepath)

    tr_samples = dataset.get_data('train')[0]
    str_tr_samples = list(set([str(sample) for sample in tr_samples]))
    tr_status = data.check(str_tr_samples, nr_clusters +
                           2 * nr_clusters * NR_PCA_COMPONENTS)

    te_samples = dataset.get_data('test')[0]
    str_te_samples = list(set([str(sample) for sample in te_samples]))
    te_status = data.check(str_te_samples, nr_clusters +
                           2 * nr_clusters * NR_PCA_COMPONENTS)

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


def main():
    try:
        opt_pairs, args = getopt.getopt(
            sys.argv[1:], "hd:k:t:o:",
            ["help", "dataset=", "nr_clusters=", "task=", "out_folder="])
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
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-o", "--out_folder"):
            kwargs["outfolder"] = arg

    if task not in ("check", "merge"):
        print "Unknown task."
        usage()
        sys.exit(1)

    if task == "check":
        check_given_dataset(src_cfg, nr_clusters)
    elif task == "merge":
        merge_given_dataset(src_cfg, nr_clusters, **kwargs)


if __name__ == '__main__':
    main()
