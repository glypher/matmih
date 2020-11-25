"""hyperparameters.py: Helper class to process images
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import itertools
from shutil import copyfile
import uuid

from .model import Model, ModelHistorySet
from .features import ModelDataSet


class HyperParamsLookup:
    def __init__(self, model, performance_callback):
        self._model = model
        self._best_history = None
        self._best_checkpoint = './best_model_' + str(uuid.uuid4()) + '.save'
        self._history = ModelHistorySet()
        self._models = []
        self._best_model = None
        self._best_performance = -1e10
        self._performance_callback = performance_callback

    def grid_search(self, data: ModelDataSet, log=False, destroy_model=True, save_best=False, **hyper_space):
        hyper_keys = hyper_space.keys()
        hyper_values = hyper_space.values()

        for hyper_params in itertools.product(*hyper_values):
            model_init = {}
            for hyper_key, hyper_val in zip(hyper_keys, hyper_params):
                model_init[hyper_key] = hyper_val

            if isinstance(self._model, Model):
                model = self._model
                model.__init__(**model_init)
            else:
                model = self._model(model_init)

            history = model.train(data)

            self._history.add_history(history)
            perf = self._performance_callback(history)
            if log:
                print("Hyperparameters: {0}\nResults: {1}".format(model_init, perf))
            if perf > self._best_performance:
                self._best_performance = perf
                self._best_history = history
                if model.checkpoint() is not None:
                    copyfile(model.checkpoint(), self.best_checkpoint)

                if save_best:
                    if self._best_model and destroy_model:
                        self._best_model.destroy()
                    self._best_model = model

            if destroy_model and self._best_model != model:
                model.destroy()
            else:
                self._models.append(model)

    @property
    def best_params(self):
        return self._best_history.model_params

    @property
    def best_history(self):
        return self._best_history

    @property
    def best_model(self):
        return self._best_model

    @property
    def history(self):
        return self._history

    @property
    def models(self):
        return self._models

    @property
    def best_checkpoint(self):
        return self._best_checkpoint
