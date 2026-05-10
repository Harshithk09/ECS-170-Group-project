'''
Dataset loader for CIFAR-10 (32x32x3, 10 classes). Expects a pickle file with the
course format: {"train": [{"image": array, "label": int}, ...], "test": [...]}.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_CIFAR10(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    preloaded_data = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    @staticmethod
    def course_dict_from_torchvision(root_dir):
        """Download CIFAR-10 via torchvision and return the course pickle-style dict."""
        from torchvision.datasets import CIFAR10

        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        train_ds = CIFAR10(root, train=True, download=True)
        test_ds = CIFAR10(root, train=False, download=True)
        train_X = [np.asarray(img) for img, _ in train_ds]
        train_y = [int(y) for _, y in train_ds]
        test_X = [np.asarray(img) for img, _ in test_ds]
        test_y = [int(y) for _, y in test_ds]
        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y},
        }

    def load(self):
        if self.preloaded_data is not None:
            print('CIFAR-10 LOADING (preloaded)')
            return self.preloaded_data
        print('CIFAR-10 LOADING')
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
