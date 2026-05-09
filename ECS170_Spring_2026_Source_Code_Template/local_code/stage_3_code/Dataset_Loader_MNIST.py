
#Dataset class for the MNIST digit image dataset

import pickle
from local_code.base_class.dataset import dataset

class Dataset_Loader_MNIST(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName = None, dDescription = None):
        super().__init__(dName, dDescription)

        def load(self):
                print("MNIST LOADING")
                #This is where we open pickle file
                #MNIST -> binary pickle with a dict:
                f = open(self.dataset_source_folder_path +
                         self.dataset_source_file_name, 'rb')
                raw =pickle.load(f)
                f.close()

                #Training split
                train_X = []
                train_y = []
                for instance in raw['train']:

                    #instance[image] -> 2d np arr (28, 28)
                    train_X.append(instance['image'])
                    train_y.append(instance['label']) #0-9
                test_X = []
                test_y = []
                for instance in raw['test']:
                    test_X.append(instance['image'])
                    test_y.append(instance['label'])
                return{
                    'train' : {'X': train_X, 'y': train_y},
                    'test': {'X': test_X, 'y': test_y}
                }




