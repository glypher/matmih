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
        self._model_inits = []
        self._best_model = None
        self._best_performance = -1e10
        self._checkpoints = []
        self._performance_callback = performance_callback

    def grid_search(self, data: ModelDataSet, log=False, destroy_model=True, save_checkpoints=False, save_best=False, **hyper_space):
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

            if save_checkpoints and model.checkpoint() is not None:
                self._checkpoints.append(model.checkpoint())

            if destroy_model and self._best_model != model:
                model.destroy()
            else:
                self._models.append(model)

    def parallel_grid_search(self, data: ModelDataSet, **hyper_space):
        hyper_keys = hyper_space.keys()
        hyper_values = hyper_space.values()

        for hyper_params in itertools.product(*hyper_values):
            model_init = {}
            for hyper_key, hyper_val in zip(hyper_keys, hyper_params):
                model_init[hyper_key] = hyper_val

            self._model_inits.append((data, model_init.copy()))

    def parallel_execute(self, log=False, num_threads=2, save_checkpoints=False):
        import concurrent.futures

        def _train_model(model_init):
            data, model_init = model_init
            model = self._model(model_init)
            history = model.train(data)
            history.model_params['checkpoint'] = model.checkpoint()
            if log:
                print(f"Hyperparameters: {history.model_params}\nResults: {self._performance_callback(history)}")
            model.destroy()
            return history

        histories = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_train_model, model_init) for model_init in self._model_inits]
            for future in concurrent.futures.as_completed(futures):
                try:
                    histories.append(future.result())
                except Exception as e:
                    print(f"Exception encountered: {e}")

            executor.shutdown()

        for history in histories:
            self._history.add_history(history)

            perf = self._performance_callback(history)
            if perf > self._best_performance:
                self._best_performance = perf
                self._best_history = history
                if history.model_params['checkpoint'] is not None:
                    copyfile(history.model_params['checkpoint'], self.best_checkpoint)

            if save_checkpoints and history.model_params['checkpoint'] is not None:
                self._checkpoints.append(history.model_params['checkpoint'])

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

    @property
    def checkpoints(self):
        return self._checkpoints
