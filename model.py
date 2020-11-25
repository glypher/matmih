"""model.py: Base class for ML models
Common API for models regardless of their framework implementation
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

from abc import ABC, abstractmethod
import numpy as np
import random
import sklearn
from datetime import datetime
from enum import Enum

from .features import ModelDataSet


class Model(ABC):
    @staticmethod
    def accuracy(true_labels, predicted_labels):
        return np.sum(true_labels == predicted_labels) / len(true_labels)

    @staticmethod
    def r2_score(true_values, predicted_values):
        import sklearn
        return sklearn.metrics.r2_score(true_values, predicted_values)

    @staticmethod
    def mse(true_values, predicted_values):
        import sklearn
        return sklearn.metrics.mean_squared_error(true_values, predicted_values)

    @abstractmethod
    def train(self, data_set: ModelDataSet, log=False):
        """Trains a model and checks validation
        """
        pass

    @abstractmethod
    def predict(self, features):
        """
        Returns a tuple of predicted labels and confidence levels for those predictions
        """
        pass

    @abstractmethod
    def checkpoint(self):
        """
        Creates a checkpoint of the best model
        """
        return None

    @abstractmethod
    def destroy(self):
        """
        Clears the model
        """
        pass


class DataType(Enum):
    TRAIN = 0,
    VALIDATION = 1,
    TEST = 2


class ModelHistory:
    def __init__(self, model_params, history):
        self._model_params = model_params.copy()
        self._history = history.copy()

    @property
    def model_params(self):
        return self._model_params

    def history(self, metric, type : DataType):
        try:
            if type == DataType.TRAIN:
                return self._history[metric]
            if type == DataType.VALIDATION:
                return self._history['val_' + metric]
            if type == DataType.TEST:
                return self._history['test_' + metric]
        except:
            return None


class ModelHistorySet:
    def __init__(self):
        self._histories = []

    def add_history(self, history: ModelHistory):
        self._histories.append(history)

    @property
    def histories(self):
        return self._histories

    def filter_histories(self, **params):
        histories = []
        for h in self._histories:
            if params.items() <= h.model_params.items():
                histories.append(h)

        return histories

    def same_histories(self, params: list):
        histories = {}
        for h in self._histories:
            p = ''.join(['{}={} '.format(k, h.model_params[k]) for k in params])
            same = histories.get(p, [])
            same.append(h)
            histories[p] = same

        return histories


class RandomClassifier(Model):
    def __init__(self, **hyper_params):
        self.__epochs = hyper_params['epochs']
        self._classes = None
        self._target_prob = None
        pass

    def train(self, data_set: ModelDataSet, balanced=True):
        random.seed(datetime.now())
        self._classes = data_set.classes

        # compute the class probabilities
        if balanced:
            class_values = sklearn.utils.class_weight.compute_class_weight('balanced',
                                                                           data_set.classes, data_set.train_target)
            class_values = 1 / class_values
            self._target_prob = sklearn.utils.extmath.softmax([class_values])[0]
        else:
            self._target_prob = np.full((len(data_set.classes,),), 1 / len(data_set.classes))

        train_acc = 0
        val_acc = 0
        train_loss = 0
        val_loss = 0
        for _ in range(self.__epochs):
            random_train = [np.random.choice(len(data_set.classes), p=self._target_prob)
                            for i in range(len(data_set.train_target))]
            random_validation = [np.random.choice(len(data_set.classes), p=self._target_prob)
                                 for i in range(len(data_set.validation_target))]

            train_error = np.sum(random_train == data_set.train_target)
            val_error = np.sum(random_validation == data_set.validation_target)

            train_acc += train_error / len(data_set.train_target)
            val_acc += val_error / len(data_set.validation_target)
            train_loss += train_error
            val_loss += val_error

        return {
            'accuracy': train_acc / self.__epochs,
            'loss': val_acc / self.__epochs,
            'val_accuracy': val_acc / self.__epochs,
            'val_loss': val_loss / self.__epochs
        }

    def predict(self, features):
        targets = np.array([np.random.choice(len(self._classes), p=self._target_prob) for i in range(len(features))])
        scores = self._target_prob
        return targets, scores

    def checkpoint(self):
        return None

    def destroy(self):
        pass
