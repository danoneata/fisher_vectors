from .base_evaluation import BaseEvaluation

def Evaluation(type_evaluation):
    for cls in BaseEvaluation.__subclasses__():
        if cls.is_evaluation_for(type_evaluation):
            return cls()
    raise ValueError
