import os
import sys

from numpy import dot, sum, zeros

from fv_utils import normalize, standardize

class BaseModel(object):
    """ Interface for the model class.

    """
    def __init__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def get_features(self, dataset):
        pass

    def evaluate(self, dataset):
        pass

class BOWModel(BaseModel):
    """ Implementation for bag of words model.

    """
    def __init__(self, K):
        self.K = K

    def __str__(self):
        print 'Bag of words model.'

    def get_kernel_matrix(self, dataset):
        """Computes the kernel matrices that are needed for the evaluation.

        Input:
            dataset     instance of the class Dataset

        Output:
            Kxx         (NxN) kernel matrix of the training samples
            Kyx         (MxN) kernel matrix between the testing samples and the
                        the training samples
            Cx          (NxC) binary matrix that specifies the labels of the 
                        train data
            Cy          (MxC) binary matrix that specifies the labels of the
                        test data.

        """
        train_features = self._get_features_set(dataset, 'train')
        test_features = self._get_features_set(dataset, 'test')

        Kxx = dot(train_features, train_features)
        Kyx = dot(test_features, train_features)
        # Build corresponding class label matrices.
        return Kxx, Cx, Kyx, Cy

    def get_features(self, dataset, set, FLG_STD=True, FLG_PN=True):
        """Prepares the features for the bag of words model. 

        Input:
            dataset     instance of the class Dataset
            set         specifies whether to load the train set features or 
                        the test set ones -- can be 'train' or 'test'
            FLG_STD     flag for standardization
            FLG_PN      flag for power normalization.

        Output:
            features    (NxK) matrix of the features. 

        """
        assert set in ['train', 'test']
        samples = dataset.get_data('train')[0]
        N = len(samples)
        features = zeros(N, K)
        for ii, sample in enumerate(samples):
            stats = get_sample_statistics(sample)
            features[ii] = sum(stats[:,:K])
        if FLG_STD:
            # Standardization
            if set == 'train': 
                features, mu, std = standardize(features)
                # save mu and std.
            else:
                features = standardize(features, mu, std)
        if FLG_PN: 
            features = power_normalize(features)
        return features

    @classmethod
    def is_model_for(cls, app_model, sp_model):
        if app_model == 'bow' and sp_model == None:
            return True
        else:
            return False

class FVModel(BaseModel):
    """ Implementation for Fisher vectors model.

    """
    def __init__(self, K):
        self.K = K

    def __str__(self):
        print 'Fisher vector model.'

    def get_features(self, dataset):
        pass

    @classmethod
    def is_model_for(cls, app_model, sp_model):
        if app_model == 'fv' and sp_model == None:
            return True
        else:
            return False

class BOWSPMModel(BaseModel):
    """ Implementation for the combination bag of words + spatial pyramid 
    matching.

    """
    def __init__(self, K, grids):
        self.K = K
        self.grids = grids
        
    def __str__(self):
        print 'Bag of words + spatial pyramids.'

    @classmethod
    def is_model_for(cls, app_model, sp_model):
        if app_model == 'bow' and sp_model == 'spm':
            return True
        else:
            return False

def Model(app_model, K, sp_model=None, sp_param=None):
    for cls in BaseModel.__subclasses__():
        if cls.is_model_for(app_model, sp_model):
            return cls(app_model, K, sp_model, sp_param)
    raise ValueError
