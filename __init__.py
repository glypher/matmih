"""
Helper library for Practical Machine Learning course
Mihai Matei - Data Science master @2019
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"
__version__ = "0.0.1"

from .plot import PlotBuilder
from .model import Model, DataType, ModelHistory, ModelHistorySet, RandomClassifier
from .tensorflow import TensorBoard, TensorModel
from .sklearn import SklearnModel
from .image import Image, ImageGenerator
from .data import DataSet
from .features import ModelDataSet, DataModel
from .hyperparameters import HyperParamsLookup
from .evaluation import ModelEvaluation

__all__ = [
    'PlotBuilder',
    'Model',
    'DataType',
    'ModelHistory',
    'ModelHistorySet',
    'RandomClassifier',
    'TensorBoard',
    'TensorModel',
    'SklearnModel',
    'Image',
    'ImageGenerator',
    'DataSet',
    'ModelDataSet',
    'DataModel',
    'HyperParamsLookup',
    'ModelEvaluation'
]
