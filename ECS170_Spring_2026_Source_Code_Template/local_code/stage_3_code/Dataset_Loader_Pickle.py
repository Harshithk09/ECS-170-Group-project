'''
Loads Stage 3 image pickles in the course layout (same as script_data_loader.py / loader.py):

  raw = pickle.load(open('ORL'|'CIFAR'|'MNIST', 'rb'))
  raw['train'] / raw['test']  ->  list of {'image': ndarray, 'label': int}

Returns the template structure expected by settings and methods:

  {'train': {'X': [...], 'y': [...]}, 'test': {'X': [...], 'y': [...]}}

Set dataset_source_folder_path (trailing slash optional) and dataset_source_file_name
(e.g. 'ORL', 'CIFAR', 'MNIST').
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_Pickle(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    preloaded_data = None
    # ORL is stored as HxWx3 with identical channels; use one channel for a single-channel CNN.
    orl_use_first_channel_only = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _normalize_image(self, image):
        arr = np.asarray(image)
        key = (self.dataset_source_file_name or '').strip().upper()
        is_orl = key == 'ORL' or key.startswith('ORL.')
        if (
            self.orl_use_first_channel_only
            and is_orl
            and arr.ndim == 3
            and arr.shape[2] >= 1
        ):
            return arr[:, :, 0].copy()
        return arr

    def load(self):
        if self.preloaded_data is not None:
            print('Pickle dataset LOADING (preloaded)')
            return self.preloaded_data

        folder = self.dataset_source_folder_path or ''
        if folder and not folder.endswith('/') and not folder.endswith('\\'):
            folder = folder + '/'
        path = folder + (self.dataset_source_file_name or '')
        print(f'Pickle dataset LOADING: {path}')

        with open(path, 'rb') as f:
            raw = pickle.load(f)

        train_X, train_y = [], []
        for instance in raw['train']:
            train_X.append(self._normalize_image(instance['image']))
            train_y.append(instance['label'])

        test_X, test_y = [], []
        for instance in raw['test']:
            test_X.append(self._normalize_image(instance['image']))
            test_y.append(instance['label'])

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y},
        }
