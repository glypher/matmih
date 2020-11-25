"""tensorflow.py: Utility scikit-learn classes
Contains a genric model for sklearn
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import os
import pickle
from abc import abstractmethod
import uuid

from .model import Model
from .features import ModelDataSet


class SklearnModel(Model):
    def __init__(self, model, checkpoint=True):
        self._model = model
        self._best_weights_path = './best_sklearn_' + str(uuid.uuid4()) + '.bin' if checkpoint else None

    def save_model(self, path='.', name='sk_model'):
        """ Saves a sklearn model as a python picle object dump
        """
        with open(os.path.join(path, name + '.bin'), 'wb') as pickle_file:
            pickle.dump(self._model, pickle_file)

    @staticmethod
    def load_model(path='.', name='sk_model'):
        """ Creates a sklearn model from python object load
        """
        with open(os.path.join(path, name + '.bin'), 'rb') as pickle_file:
            return SklearnModel(pickle.load(pickle_file))

    def load_weights(self, path):
        with open(path, 'rb') as pickle_file:
            self._model = pickle.load(pickle_file)

    @abstractmethod
    def train(self, data_set: ModelDataSet, log=False):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    def checkpoint(self):
        if self._best_weights_path:
            with open(self._best_weights_path, 'wb') as pickle_file:
                pickle.dump(self._model, pickle_file)
        return self._best_weights_path

    def destroy(self):
        if self._best_weights_path and os.path.exists(self._best_weights_path):
            os.remove(self._best_weights_path)
