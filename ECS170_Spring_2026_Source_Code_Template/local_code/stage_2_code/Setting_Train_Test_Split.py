'''
Stage 2 setting: use pre-split train.csv and test.csv (no random split, no shuffle).
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pandas as pd

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
        self.train_data = None
        self.test_data = None
        # Used by load_run_save_evaluate(); override before running (e.g. 'class').
        self.label_column = 'label'

    def load(self, train_path, test_path):
        """
        Load two separate CSV files instead of randomly splitting one file.
        Expects each CSV to have feature columns and a label column (see split()).
        """
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

    def split(self, label_column='label'):
        """
        Separate features and labels from each file. Returns
        (X_train, y_train, X_test, y_test) as numpy arrays.
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError('Data not loaded. Call load(train_path, test_path) first.')

        X_train = self.train_data.drop(columns=[label_column]).values
        y_train = self.train_data[label_column].values

        X_test = self.test_data.drop(columns=[label_column]).values
        y_test = self.test_data[label_column].values

        print(f'Train size : {len(X_train)} samples')
        print(f'Test size  : {len(X_test)} samples')
        print(f'Features   : {X_train.shape[1]}')

        return X_train, y_train, X_test, y_test

    def load_run_save_evaluate(self):
        """
        Template hook: expects load() already called on CSV paths, and prepare()
        already wired dataset, method, result, evaluate (dataset unused here).
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError('Call load(train_path, test_path) before load_run_save_evaluate().')

        X_train, y_train, X_test, y_test = self.split(label_column=self.label_column)
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None

    def __str__(self):
        if self.train_data is None or self.test_data is None:
            return 'Setting_Train_Test_Split: no data loaded yet.'
        return (
            f"Setting_Train_Test_Split\n"
            f"  Train samples: {len(self.train_data)}\n"
            f"  Test samples : {len(self.test_data)}"
        )
