'''
Concrete MNIST dataset loader for Stage 3 CNN
'''

from local_code.base_class.dataset import dataset
import pickle


class Dataset_Loader_MNIST(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading MNIST data...')

        file_path = self.dataset_source_folder_path + self.dataset_source_file_name

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        train_X = []
        train_y = []
        test_X = []
        test_y = []

        for instance in data['train']:
            train_X.append(instance['image'])
            train_y.append(instance['label'])

        for instance in data['test']:
            test_X.append(instance['image'])
            test_y.append(instance['label'])

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y}
        }