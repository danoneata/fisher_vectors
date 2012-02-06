#!/usr/bin/python

import getopt
import os
import sys 
import multiprocessing as mp
from ipdb import set_trace

from dataset import Dataset
from fisher_vectors.features import DescriptorProcessor
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.model import Model

ROOT_PATH = '/home/clear/oneata/data'
RESULTS_FN = 'new_results.txt'

def do(task, src_cfg, ip_type, model_type, K, grid, Ncpus, verbose):
    dataset = Dataset(src_cfg, ip_type=ip_type)
    model = Model(model_type, K, grid)
    dp = DescriptorProcessor(dataset, model)
    # Select the task based on opt_type argument.
    if task == 'compute':
        dp.compute_statistics(Ncpus)
    elif task == 'merge':
        dp.merge_statistics()
    elif task == 'remove':
        dp.remove_statistics()
    elif task == 'evaluate':
        if dataset.DATASET == 'kth':
            evn = 'svm_one_vs_all'
        elif dataset.DATASET == 'hollywood2':
            evn = 'svm_one_vs_one'
        evaluation = Evaluation(evn)
        model.fit(dataset, evaluation)
        with open(os.path.join(ROOT_PATH, dataset.DATASET, 'results', RESULTS_FN), 'a') as ff:
            ff.write('%s %2.3f\n' % (str(model), 100 * model.score()))
    else:
        raise Exception("Option type is not defined.")
        usage()
        sys.exit(1)

def usage():
    prog_name = os.path.basename(sys.argv[0])
    print ('Usage: %s opt_type -d dataset -i ip_type -m model -k K [-g grid] [-n ncpus]'
           % prog_name)
    print
    print 'Launches different tasks for a given dataset and model.'
    print
    print 'Options:'
    print '     -o, --opt_type=OPTION_TYPE'
    print '         Specify the task:'
    print '             - "compute": computes statistics'
    print '             - "merge": merge statistics'
    print '             - "remove": removes statistics'
    print '             - "evaluate": evaluates the model.'
    print
    print '     -d, --dataset=DATASET'
    print '         Specify the dataset to be loaded (e.g., "kth" or '
    print '         "hollywood2_clean)."'
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
    print
    print '     -k, --K=K'
    print '         Specify the number of clusters used for the dictionary.'
    print 
    print '     -g --grid=GRID'
    print '         Specify whether spatial pyramids should be used. The format'
    print '         is "nx_ny_nt", where "n[xyt]" denotes the number of cells'
    print '         that are used for the corresponding dimension (horizontal,'
    print '         vertical, or temporal). By default, there is no spatial'
    print '         pyramids used (i.e., we use "1_1_1").'
    print 
    print '     -n, --ncpus=NCPUS'
    print '         Number of cores to run the operations on. By default, this'
    print '         is set to the number of nodes on the cluster.'
    print 
    print '     Other options: -h, --help and -v, --verbose (these do not'
    print '     require any argument).'

def main():
    try:
        task = sys.argv[1]
        opt_pairs, _args = getopt.getopt(sys.argv[2:], "hvd:i:m:k:g:n:", ["help", "verbose", "dataset=", "ip_type=", "model=", "K=", "grid=", "ncpus="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(1)
    except IndexError:
        print 'Not enough arguments.'
        usage()
        sys.exit(1)
    verbose = False 
    Ncpus = mp.cpu_count()
    grid = [(1, 1, 1)]
    for opt, arg in opt_pairs:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-d", "--dataset"):
            src_cfg = arg
        elif opt in ("-i", "--ip_type"):
            ip_type = arg
        elif opt in ("-m", "--model"):
            model_type = arg
        elif opt in ("-k", "--K"):
            K = int(arg)
        elif opt in ("-g", "--grid"):
            gx, gy, gt = arg.split('_')
            grid = [(int(gx), int(gy), int(gt))]
        elif opt in ("-n", "--ncpus"):
            Ncpus = min(max(int(arg),1), Ncpus)
    do(task, src_cfg, ip_type, model_type, K, grid, Ncpus, verbose)
    
if __name__ == '__main__':
    main()
