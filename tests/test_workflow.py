from dataset import Dataset
from fisher_vectors.features import DescriptorProcessor
from fisher_vectors.model import Model
from fisher_vectors.evaluation import Evaluation

K = 100
ds = Dataset('hollywood2_clean')
grid_0 = (1, 1, 1)
grid_1 = (1, 3, 1)
grid_2 = (1, 1, 2)
if ds.DATASET == 'kth':
    evn = 'svm_one_vs_all'
elif ds.DATASET == 'hollywood2':
    evn = 'svm_one_vs_one'

def test_descriptor_processor():
    md = Model('fv_sfv', K)
    dp = DescriptorProcessor(ds, md)
    dp.compute_statistics()
    dp.grid = grid_1
    dp.compute_statistics()
    dp.grid = grid_2
    dp.compute_statistics()

def test_model():
    md = Model('fv_sfv', K, [grid_0, grid_1, grid_2])
    ev = Evaluation(evn)
    md.fit(ds, ev)
    print md.score()

def test_model_evaluation():
    md = Model('fv', K, [grid_0, grid_1, grid_2])
    ev = Evaluation(evn)
    md.fit(ds, ev)
    print md.score()

def test_bow_model():
    md = Model('bow', K, [grid_0])
    ev = Evaluation(evn)
    md.fit(ds, ev)
    print md.score()

def main():
    test_descriptor_processor()
    test_model_evaluation()
    #test_bow_model()
    #test_model()
   
if __name__ == '__main__':
    main()
