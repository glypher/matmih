"""features.py: Helper classes to hold dataset information
Process a data set to be used with a Model class
The features and targets are computed as numpy darrays
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import numpy as np
from .data import DataSet

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split


class ModelDataSet:
    def __init__(self, data_set: DataSet):
        self._data_set = data_set
        self._train_features = None
        self._train_target = None
        self._validation_features = None
        self._validation_target = None
        self._test_features = None
        self._test_target = None
        self._feature_shape = None
        self._norm_transform = None
        self._constant_filter = None
        self._anova_filter = None

    def flatten(self):
        self._feature_shape = self.train_features.shape[1:]
        self._train_features = self.train_features.reshape((self.train_features.shape[0], -1))
        if self.validation_features is not None:
            self._validation_features = self.validation_features.reshape((self.validation_features.shape[0], -1))
        if self.test_features is not None:
            self._test_features = self.test_features.reshape((self.test_features.shape[0], -1))

        return self

    def un_flatten(self):
        self._train_features = self.train_features.reshape(
            (self.train_features.shape[0], *self._feature_shape))
        if self.validation_features is not None:
            self._validation_features = self.validation_features.reshape(
                (self.validation_features.shape[0], *self._feature_shape))
        if self.test_features is not None:
            self._test_features = self.test_features.reshape(
                (self.test_features.shape[0], *self._feature_shape))

        return self

    def unflatten(self, features):
        return features.reshape((features.shape[0], *self._feature_shape))

    def augment_train(self, train_features, train_target):
        self._train_features = np.concatenate((self.train_features, train_features), axis=0)
        self._train_target = np.concatenate((self.train_target, train_target), axis=0)

        return self

    def normalize(self, mean=0, std=1):
        self._norm_transform = StandardScaler()

        # Fit the scale transformer on the training data
        self._norm_transform.mean_ = np.full((self.train_features.shape[1],), mean)
        self._norm_transform.var_ = np.full((self.train_features.shape[1],), std**2)

        self._train_features = self._norm_transform.fit_transform(self.train_features)
        if self.validation_features is not None:
            self._validation_features = self._norm_transform.transform(self.validation_features)
        if self.test_features is not None:
            self._test_features = self._norm_transform.transform(self.test_features)

        return self

    def unormalize(self, features):
        return self._norm_transform.inverse_transform(features)

    def normalize_MobileNetV2(self):
        '''
        Special normalization for MobileNet2
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
        https://arxiv.org/abs/1801.04381
        '''
        self._train_features = self.train_features / 128. - 1.
        if self.validation_features is not None:
            self._validation_features = self.validation_features / 128. - 1.
        if self.test_features is not None:
            self._test_features = self.test_features / 128. - 1.

        return self

    def filter(self, max_features=100):
        assert max_features is None or max_features <= self.train_features.shape[1]

        self._constant_filter = VarianceThreshold(threshold=0.1)
        self._train_features = self._constant_filter.fit_transform(self.train_features)
        if self.validation_features is not None:
            self._validation_features = self._constant_filter.transform(self.validation_features)
        if self.test_features is not None:
            self._test_features = self._constant_filter.transform(self.test_features)

        self._anova_filter = SelectKBest(f_classif, k=max_features)
        self._train_features = self._anova_filter.fit_transform(self.train_features, self.train_target)
        if self.validation_features is not None:
            self._validation_features = self._anova_filter.transform(self.validation_features)
        if self.test_features is not None:
            self._test_features = self._anova_filter.transform(self.test_features)

        return self

    def unfilter(self, features):
        return self._constant_filter.inverse_transform( self._anova_filter.inverse_transform(features) )

    @property
    def train_features(self):
        if self._train_features is None:
            self._train_features = self._data_set.train_features.to_numpy()
            self._train_features = np.stack(self._train_features, axis=0)
        return self._train_features

    @property
    def train_target(self):
        if self._train_target is None:
            try:
                self._train_target = self._data_set.train_target.cat.codes.to_numpy()
            except AttributeError:
                self._train_target = self._data_set.train_target.to_numpy()
        return self._train_target

    @property
    def validation_features(self):
        if self._validation_features is None:
            if self._data_set is None or self._data_set.validation_features is None:
                return None
            self._validation_features = self._data_set.validation_features.to_numpy()
            self._validation_features = np.stack(self._validation_features, axis=0)
        return self._validation_features

    @property
    def validation_target(self):
        if self._validation_target is None:
            if self._data_set is None or self._data_set.validation_target is None:
                return None
            try:
                self._validation_target = self._data_set.validation_target.cat.codes.to_numpy()
            except AttributeError:
                self._validation_target = self._data_set.validation_target.to_numpy()
        return self._validation_target

    @property
    def test_features(self):
        if self._test_features is None:
            if self._data_set is None or self._data_set.test_features is None:
                return None
            self._test_features = self._data_set.test_features.to_numpy()
            self._test_features = np.stack(self._test_features, axis=0)
        return self._test_features

    @property
    def test_target(self):
        if self._test_target is None:
            if self._data_set is None or self._data_set.test_target is None:
                return None
            try:
                self._test_target = self._data_set.test_target.cat.codes.to_numpy()
            except AttributeError:
                self._test_target = self._data_set.test_target.to_numpy()
        return self._test_target

    @property
    def classes(self):
        return self._data_set.class_ids


class DataModel(ModelDataSet):
    def __init__(self, features, target):
        super(DataModel, self).__init__(None)
        self._all_features = features
        self._all_target = target

    def split(self, split: list):
        x, val_x, y, val_y = train_test_split(self._all_features, self._all_target,
                                              train_size=split[0], test_size=split[1])
        self.set_train(x, y)
        self.set_validation(val_x, val_y)
        return self

    def get_kfold_split(self, kfold=5):
        kf = KFold(n_splits=kfold, shuffle=False)
        for train, validation in kf.split(self._all_features, self._all_target):
            self.set_train(self._all_features[train], self._all_target[train])
            self.set_validation(self._all_features[validation], self._all_target[validation])
            yield self

    def set_train(self, features, target):
        self._train_features = features
        self._train_target = target
        return self

    def set_validation(self, features, target):
        self._validation_features = features
        self._validation_target = target
        return self

    def set_test(self, features, target):
        self._test_features = features
        self._test_target = target
        return self

    @property
    def features(self):
        return self._all_features

    @property
    def targets(self):
        return self._all_target