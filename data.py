import cPickle
import numpy as np
import os


class SstatsMap(object):
    """ Wrapper for the temporary data file, sufficient statistics. """
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
