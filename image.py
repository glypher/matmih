"""image.py: Helper class to process images
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import numpy as np
import sklearn
import tensorflow as tf
from PIL import Image as PImage


from .data import DataSet


class Image:
    @staticmethod
    def load(path, size=False, fill_color = (255, 255, 255)):
        if path.split('.')[-1] not in ('png', 'jpg'):
            raise Exception('Not a png/jpg image!')

        img = None
        with PImage.open(path) as img:
            img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA') or size != False:
                size = img.size if size == False else size
                temp_img = PImage.new(img.mode[:-1], size, fill_color)
                temp_img.paste(img, img.split()[-1]) # omit transparency
                img = temp_img

        return np.array(img, dtype=np.uint8)

    @staticmethod
    def to_image(data):
        return PImage.fromarray(data)


class ImageGenerator:
    def __init__(self, data_set: DataSet, balanced=False, **kwargs):
        self._data_set = data_set
        self._image_generator = tf.keras.preprocessing.image.ImageDataGenerator(**kwargs)

        # Fit the image generator on the training set
        self._image_generator.fit(self._data_set.train_features)

        # compute the class probabilities
        self._balanced = balanced
        if self._balanced:
            class_values = sklearn.utils.class_weight.compute_class_weight('balanced',
                                                                           data_set.classes, data_set.train_target)
            self._target_prob = sklearn.utils.extmath.softmax([class_values])[0]

    def generate(self, iterations=100, batch_size=32):
        while iterations > 0:
            iterations -= 1
            if self._balanced:
                # take 3 times more batches
                image_b, target_b = next(self._image_generator.flow(self._data_set.train_features,
                                                                    self._data_set.train_target,
                                                                    batch_size=3 * batch_size))
                # select targets from the larger batch with probabilities taken from the target distribution
                image = np.zeros((batch_size, *self._data_set.train_features.shape[1:]))
                target = np.zeros(batch_size)
                for i in range(batch_size):
                    cid = None
                    img_id = -1
                    while img_id < 0:
                        cid = np.random.choice(len(self._data_set.classes), p=self._target_prob)
                        img_id = np.argwhere(target_b==cid)
                        img_id = img_id[0] if len(img_id) > 0 else -1

                    target_b[img_id] = -1
                    target[i] = cid
                    image[i] = image_b[img_id]
            else:
                image, target = next(self._image_generator.flow(self._data_set.train_features,
                                                                self._data_set.train_target,
                                                                batch_size=batch_size))

            yield image.astype(np.uint8), target.astype(np.uint8)
