"""tensorflow.py: Utility tensor flow classes
Contains callbacks useful for tensorboard and more
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import io
import os
import subprocess
import tensorflow as tf
import uuid
from abc import abstractmethod

from .plot import PlotBuilder
from .model import Model
from .features import ModelDataSet


class TensorModel(Model):
    def __init__(self, model, checkpoint=True):
        self._model = model
        self._best_weights_path = './model_checkpoint/best_epoch_weights_' + str(uuid.uuid4()) + '.h5' if checkpoint else None

    def save_model(self, path='./model_checkpoint', name='tf_model'):
        """ Saves a tensor flow model configuration and weights
        """
        json_config = self._model.to_json()
        with open(os.path.join(path, name + '.json'), 'w') as json_file:
            json_file.write(json_config)
        weights_path = os.path.join(path, name + '_weights.h5')
        self._model.save_weights(weights_path)

    @staticmethod
    def load_model(path='./model_checkpoint', name='tf_model'):
        """ Creates a tensor flow model from configuration and weights
        """
        with open(os.path.join(path, name + '.json')) as json_file:
            json_config = json_file.read()
        model = TensorModel(tf.keras.models.model_from_json(json_config))
        model._model.load_weights(os.path.join(path, name + '_weights.h5'))

        return model

    def load_weights(self, weights_path):
        self._model.load_weights(weights_path)

    @abstractmethod
    def train(self, data_set: ModelDataSet, log=False):
        pass

    @property
    def best_weights_path(self):
        return self._best_weights_path

    def predict(self, features):
        features_ds = tf.cast(features, tf.float32)
        return self.model.predict_classes(features_ds), self.model.predict(features_ds)

    def checkpoint(self):
        return self._best_weights_path

    def destroy(self):
        tf.keras.backend.clear_session()
        if self._model is not None:
            del self._model
            self._model = None


class TensorBoard:
    __instance = None
    LOG_DIR = './logs'

    def __init__(self, model: Model, ds_validation, classes):
        self.__model = model
        self.__ds_validation = ds_validation
        self.__classes = classes
        TensorBoard.__instance = self

    @staticmethod
    def get_log_dir(log_dir=LOG_DIR):
        import datetime
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @staticmethod
    def open(log_dir=LOG_DIR):
        try:
            return subprocess.Popen(["tensorboard", "--logdir", "{{{}}}".format(log_dir)],
                                    stdout=subprocess.PIPE)
        except Exception as e:
            print(e)

    @staticmethod
    def callback_confusion_matrix(epoch, logs):
        """
        Called by tensorflow/keras training layer on each epoch
        """
        if TensorBoard.__instance is None:
            return
        self = TensorBoard.__instance

        # Use the model to predict the values from the validation dataset.
        predictions = []
        labels = []
        for batchFeatures, batchLabels in self._validationDS:
            predictions.append(self.__model.predict(batchFeatures))
            labels.append(batchLabels)

        plot_builder = PlotBuilder()
        plot_builder.create_confusion_matrix(labels, predictions, self.__classes)

        buf = io.BytesIO()
        plt = plot_builder.get_plot()
        plt.savefig(buf, format='png')
        plot_builder.close()

        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        file_writer = tf.summary.create_file_writer(self.__log_dir + '/cm')
        # Log the confusion matrix as an image summary.
        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)
