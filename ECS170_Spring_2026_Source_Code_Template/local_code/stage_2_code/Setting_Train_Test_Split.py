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
        # Stage 2 MNIST CSV files have no header row and store the label in the
        # first column, so column index 0 is the default label column.
        self.label_column = 0
        # The provided Stage 2 data files do not include a header row.
        self.has_header = False

    def load(self, train_path, test_path):
        """
        Load two separate CSV files instead of randomly splitting one file.
        The provided Stage 2 MNIST files are headerless CSVs where the first
        column is the label and the remaining 784 columns are pixel features.
        """
        header = 0 if self.has_header else None
        self.train_data = pd.read_csv(train_path, header=header)
        self.test_data = pd.read_csv(test_path, header=header)

    def split(self, label_column=0):
        """
        Separate features and labels from each file. Returns
        (X_train, y_train, X_test, y_test) as numpy arrays.
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError('Data not loaded. Call load(train_path, test_path) first.')

        # Support either integer column indices (used by the provided MNIST
        # files) or explicit column names if the setting is reused later.
        if isinstance(label_column, int):
            X_train = self.train_data.drop(columns=[label_column]).values
            y_train = self.train_data.iloc[:, label_column].values

            X_test = self.test_data.drop(columns=[label_column]).values
            y_test = self.test_data.iloc[:, label_column].values
        else:
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
        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        # Save a concise summary that is suitable for a text report file.
        self.result.data = {
            'metrics': {key: round(value, 4) for key, value in metrics.items()},
        }
        self.result.save()
        return metrics, None

    def __str__(self):
        if self.train_data is None or self.test_data is None:
            return 'Setting_Train_Test_Split: no data loaded yet.'
        return (
            f"Setting_Train_Test_Split\n"
            f"  Train samples: {len(self.train_data)}\n"
            f"  Test samples : {len(self.test_data)}"
        )
