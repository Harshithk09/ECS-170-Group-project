'''ORL face images from the course pickle (ORL).'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader_ORL(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    use_first_channel_only = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('ORL LOADING')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            raw = pickle.load(f)

        train_X, train_y = [], []
        for instance in raw['train']:
            image = np.asarray(instance['image'])
            if self.use_first_channel_only and image.ndim == 3:
                image = image[:, :, 0]
            train_X.append(image)
            train_y.append(instance['label'])

        test_X, test_y = [], []
        for instance in raw['test']:
            image = np.asarray(instance['image'])
            if self.use_first_channel_only and image.ndim == 3:
                image = image[:, :, 0]
            test_X.append(image)
            test_y.append(instance['label'])

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y},
        }
