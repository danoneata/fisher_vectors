from .base_model import BaseModel

def Model(type_model, K, grids=[(1,1,1)]):
    """ Factory function that instantiates the object with the corresponding
    class.

    """
    classes = BaseModel.__inheritors__
    for cls in classes[BaseModel]:
        if cls.is_model_for(type_model):
            return cls(K, grids)
    raise ValueError
