'''This is the dataset for the human
face ORL images.'''

import pickle
import numpy as np
from torch.utils.data import Dataset

from local_code.base_class.dataset import dataset

class Dataset_Loader_ORL(Dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName = None, dDescription = None):
        super().__init__(dName, dDescription)
        def load(self):
            print('loading data')
            #open pickle file
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
            raw = pickle.load(f)
            f.close() #close file
            train_X = []
            train_y = []
            for instance in raw['train']:
                #(112, 92, 3) - grayscale image in RGB
                image = np.array(instance['image'])
                gray = image[:, :, 0]
                #(112, 92) R channel only
                train_X.append(gray)
                #Labels (1-40) with each representing a unique identity
                train_y.append(instance['label'])
            test_X = []
            test_y = []
            for instance in raw['test']:
                image = np.array(instance['image'])
                gray = image[:, :, 0]
                test_X.append(gray)
                test_y.append(instance['label'])
            return{
                'train' : {'X': train_X, 'y': train_y},
                'test': {'X': test_X, 'y': test_y}
            }