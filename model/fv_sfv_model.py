from ipdb import set_trace

from .fv_model import FVModel
from .sfv_model import SFVModel

class FVSFVModel(FVModel, SFVModel):
    """ Fisher vector--Spatial Fisher vector model.

    Implements functions for the model that represents both the appearance and 
    spatial information using Fisher vectors.

    Parameters
    ----------
    K: int, required
        The number of words in the vocabulary.

    grids: list of tuples, optional, default [(1, 1, 1)]
        The grids that are used to split the video for pyramid matching.

    Attributes
    ----------
    is_spatial_model: boolean
        Indicates if the current model has implemented the spatial methods.

    mm: array [1, 3], inherited
        The mean of the Gaussian.

    S: array [1, 3], inherited
        The diagonal of the covariance matrix for the Gaussian.

    Notes
    -----

    """
    def __init__(self, gmm):
        super(FVSFVModel, self).__init__(gmm)
        self.is_spatial_model = True

    def __str__(self):
        # TODO
        ss = super(FVSFVModel, self).__str__()
        return 'FV-SFV ' + ss

    def compute_kernels(self, tr_file_paths, te_file_paths,
                        tr_spatial_file_paths, te_spatial_file_paths):
        """ Computes the kernels for the FV-SFV model. It computes kernel
        products for both appearance features and spatial features. The L2
        normalization is done on the long concatenated feature vector.

        """
        super(FVSFVModel, self).compute_kernels(tr_file_paths, te_file_paths)
        self._init_normalizations()
        self._compute_kernels(tr_file_paths, te_file_paths)
        self._compute_spatial_kernels(
            tr_spatial_file_paths, te_spatial_file_paths)
        self._L2_normalize_kernels()

    @classmethod
    def is_model_for(cls, type_model):
        if type_model == 'fv_sfv':
            return True
        else:
            return False
