'''MNIST digit images from the course pickle (MNIST).'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_MNIST(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('MNIST LOADING')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            raw = pickle.load(f)

        train_X, train_y = [], []
        for instance in raw['train']:
            train_X.append(np.asarray(instance['image']))
            train_y.append(instance['label'])

        test_X, test_y = [], []
        for instance in raw['test']:
            test_X.append(np.asarray(instance['image']))
            test_y.append(instance['label'])

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y},
        }
