#!/usr/bin/python

import getopt
import os
import sys
import multiprocessing as mp
from ipdb import set_trace

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.model import Model

ROOT_PATH = '/home/clear/oneata/data'
RESULTS_FN = 'new_results.txt'


def do(task, src_cfg, **kwargs):
    ip_type = kwargs.get('ip_type', 'dense5.track15mbh')
    suffix = kwargs.get('suffix', '')
    nr_clusters = kwargs.get('nr_clusters')
    dataset = Dataset(
        src_cfg, ip_type=ip_type, suffix=suffix, nr_clusters=nr_clusters)

    per_slice = kwargs.get('per_slice', False)

    # Select the task based on opt_type argument.
    if task == 'compute':
        pass
    elif task == 'check':
        from fisher_vectors import data
        data.check_given_dataset(dataset)
    elif task == 'merge':
        if per_slice:
            from fisher_vectors.per_slice import features
            class_idx = kwargs.get('class_idx')
            features.merge_class_given_dataset(dataset, class_idx)
        else:
            from fisher_vectors import data
            data.merge_given_dataset(dataset)
    elif task == 'remove':
        pass
    elif task == 'evaluate':
        from fisher_vectors.scripts.evaluate import evaluate_given_dataset
        evaluate_given_dataset(dataset, **kwargs)
        #with open(os.path.join(
        #    ROOT_PATH, dataset.DATASET, 'results', RESULTS_FN), 'a') as ff:
        #    ff.write('%s %2.3f\n' % (str(model), 100 * model.score()))
    else:
        raise Exception("Task is not defined.")
        usage()
        sys.exit(1)


def usage():
    # TODO To be modified.
    prog_name = os.path.basename(sys.argv[0])
    print 'Usage: %s -t opt_type -d dataset -k nr_clusters \\' % prog_name
    print
    print 'Launches different tasks for a given dataset and model.'
    print
    print 'Options:'
    print '     -t, --task=OPTION_TYPE'
    print '         Specify the task:'
    print '             "compute": computes statistics'
    print '                 Parameters: '
    print '             "check": prints missing temporary statistics'
    print '             "merge": merge statistics.'
    print '                 Parameters: per_slice, class_idx'
    print '             "remove": removes statistics'
    print '             "evaluate": evaluates the model.'
    print
    print '     -d, --dataset=SRC_CFG'
    print '         Specify the configuration name of the dataset to be'
    print '         loaded (e.g., "kth").'
    print
    print '     -k, --nr_clusters=K'
    print '         Specify the number of clusters used for the dictionary.'
    print
    print '     -i, --ip_type=IP_TYPE'
    print '         Specify the type of descriptor. Default, dense5.track15mbh'
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
    print '         Default, "fv".'
    print
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
    print '     -n, --nr_processes=NCPUS'
    print '         Number of cores to run the operations on. By default, this'
    print '         is set to the number of nodes on the cluster.'
    print
    print '     --profile'
    print '         Profiles the code using cProfile. If the number of CPUs is'
    print '         set larger than 1, the profiling is done only at a'
    print '         superficial level.'
    print
    print '     Other options: -h, --help and -v, --verbose (these do not'
    print '     require any argument).'
    print
    print '     Examples:'
    print '     ---------'
    print '         fisher compute --dataset=kth --ip_type=harris.hoghof\\'
    print '         --model=fv --K=50 --ncpus=3'
    print
    print '         fisher evaluate -d hollywood2_clean -i dense5.track15mbh\\'
    print '         -m fv_sfv -k 100 -g 1_1_1-1_3_1-1_1_2'


def main():
    try:
        opt_pairs, _args = getopt.getopt(
            sys.argv[1:], "hvt:d:i:m:k:",
            ["help", "verbose", "task=", "dataset=", "ip_type=", "model=",
             "nr_clusters=", "nr_processes=", "suffix=", "per_slice",
             "class_idx=", "eval_type="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)

    kwargs = {}
    for opt, arg in opt_pairs:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-v", "--verbose"):
            kwargs['verbose'] = True
        elif opt in ("-t", "--task"):
            task = arg
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
        elif opt in ("--suffix"):
            kwargs['suffix'] = arg
        elif opt in ("--per_slice"):
            kwargs['per_slice'] = True
        elif opt in ("--class_idx"):
            kwargs['class_idx'] = int(arg)
        elif opt in("--eval_type"):
            kwargs['eval_type'] = arg

    do(task, src_cfg, **kwargs)


if __name__ == '__main__':
    main()
