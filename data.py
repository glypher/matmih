"""data.py: Helper classes to hold dataset information
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import numpy as np
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self, data_set, feature_column='features', target_column='target'):
        assert isinstance(data_set, pd.core.frame.DataFrame), 'Only panda dataframe is supported currently!'
        self.__data_set = data_set
        self._features_column = feature_column
        self._target_column = target_column
        self._train_set = None
        self._validation_set = None
        self._test_set = None
        try:
            self._classes = self.__data_set[self._target_column].cat.categories.to_numpy()
        except AttributeError:
            self._classes = []
        self._classIds = np.array(range(len(self.classes)))

    def split_data(self, splits, stratify=False, shuffle=True, augment_callback=None, filter_callback=None, sequencial=False):
        """
         Splits the data into the train, validation, test sets
        :param splits: percentage tuple found in splits (train, validation, test)
        :param stratify: if the classes density distribution should be maintained in the split,
        :param shuffle: if the data should be split randomly
        :param augment_callback: optional callback to augment the data
        :param filter_callback: optional callback to filter the data after each split
        :param sequencial: if the split must not be done randomly by in a sequencial manner |train|validation|test
        """
        train_percentage, validation_percentage, test_percentage = splits
        assert (train_percentage + validation_percentage + test_percentage == 1)

        # augment the data before the split
        data_set = self.__data_set
        if augment_callback:
            data_set = augment_callback(data_set)

        if test_percentage > 0:
            if sequencial:
                size = int((1-test_percentage) * len(data_set))
                train_val_ds, test_ds = data_set.iloc[:size], data_set.iloc[size:]
            else:
                # no class distribution on test set - we want to keep it as random as possible
                train_val_ds, test_ds = train_test_split(data_set, test_size=test_percentage,
                                                         shuffle=shuffle, random_state=int(round(time.time())),
                                                         stratify=None)

            # make sure that there is at least one of each target class in the test set
            for cl in self.classes:
                if len(test_ds[test_ds[self._target_column] == cl]) == 0:
                    idx = random.choice(train_val_ds.index[train_val_ds[self._target_column] == cl].tolist())
                    test_ds = test_ds.append(train_val_ds.loc[idx].copy(), ignore_index=True)
                    train_val_ds = train_val_ds.drop(index=idx)
            if len(self._classes) > 0:
                test_ds = test_ds.copy()
                test_ds[self._target_column] = pd.Categorical(test_ds[self._target_column],
                                                              categories=data_set[self._target_column].cat.categories)
        else:
            test_ds = None
            train_val_ds = data_set

        # filter the test and remaining set so that they do not have any features in common
        if filter_callback:
            train_val_ds, test_ds = filter_callback(train_val_ds, test_ds)

        # compute the new test percentage
        validation_percentage = validation_percentage * len(data_set) / len(train_val_ds)

        if validation_percentage > 0:
            if sequencial:
                size = int((1-validation_percentage) * len(train_val_ds))
                train_ds, validation_ds = train_val_ds.iloc[:size], train_val_ds.iloc[size:]
            else:
                # now split the train and validation set preserving the class distribution
                train_ds, validation_ds = train_test_split(train_val_ds, test_size=validation_percentage,
                                                           shuffle=shuffle, random_state=int(round(time.time())),
                                                           stratify=train_val_ds[self._target_column] if stratify else None)

            # make sure that there is at least one of each target class in the validation set
            for cl in self.classes:
                if len(validation_ds[validation_ds[self._target_column] == cl]) == 0:
                    idx = random.choice(train_ds.index[train_ds[self._target_column] == cl].tolist())
                    validation_ds = validation_ds.append(train_ds.loc[idx].copy(), ignore_index=True)
                    train_ds = train_ds.drop(index=idx)
            if len(self._classes) > 0:
                validation_ds = validation_ds.copy()
                validation_ds[self._target_column] = pd.Categorical(validation_ds[self._target_column],
                                                                    categories=data_set[self._target_column].cat.categories)
        else:
            train_ds = train_val_ds
            validation_ds = None

        # filter the test and remaining set so that they do not have any features in common
        if filter_callback:
            train_ds, validation_ds = filter_callback(train_ds, validation_ds)

        self._train_set = train_ds
        self._validation_set = validation_ds
        self._test_set = test_ds

    @property
    def train_set(self):
        return self._train_set

    @property
    def validation_set(self):
        return self._validation_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def train_features(self):
        return self._train_set[self._features_column]

    @property
    def train_target(self):
        return self._train_set[self._target_column]

    @property
    def validation_features(self):
        if self._validation_set is None:
            return None
        return self._validation_set[self._features_column]

    @property
    def validation_target(self):
        if self._validation_set is None:
            return None
        return self._validation_set[self._target_column]

    @property
    def test_features(self):
        if self._test_set is None:
            return None
        return self._test_set[self._features_column]

    @property
    def test_target(self):
        if self._test_set is None:
            return None
        return self._test_set[self._target_column]

    @property
    def classes(self):
        return self._classes

    @property
    def class_ids(self):
        return self._classIds
