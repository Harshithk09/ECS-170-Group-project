'''
Concrete CNN MethodModule for the ORL human face dataset.
Inherits from both the base method class and nn.Module (same dual-inheritance as Method_MLP).
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):

    data = None
    max_epoch = 50
    batch_size = 32
    learning_rate = 1e-3

    plot_destination_folder_path = None
    plot_file_name = 'CNN_ORL_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        #inp tensor (batch_size, 1, 112, 92)
        #ORL needs a deep network since face recognition
        #is a higher computation task

        #conv block 1 (conv2d(1->32)
        #output (batch, 32, 112, 92)
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.activation_1 = nn.ReLU()
        #maxPool: (112, 92) -> (56, 46)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv block 2 conv2d(32 -> 64)
        #output(batch, 64, 56, 46)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.activation_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv block 3
        #conv2d(64->128)
        #Output(batch, 128, 28, 23)
        self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.activation_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #FC head
        self.flatten = nn.Flatten()
        #Use dropout to avoid overfit
        self.dropout = nn.Dropout(p=0.5)
        self.fc_layer_1 = nn.Linear(128 * 14 * 11, 256)
        self.activation_4 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 40)

        self.training_history = {
            'epoch': [], 'loss': [], 'accuracy': [],
            'test_loss': [], 'test_accuracy': []
        }

    #forward pass
    def forward(self, x):
        #conv block 1
        x = self.pool_1(self.activation_1(self.conv_layer_1(x)))
        #shape: (batch, 32, 56, 46)
        #conv block 2
        x = self.pool_2(self.activation_2(self.conv_layer_2(x)))
        #shape: (batch, 64, 28, 23)
        #conv block 3
        x = self.pool_3(self.activation_3(self.conv_layer_3(x)))
        #shape: (batch, 128, 14, 11)
        #flatten
        x = self.flatten(x)
        #shape: (batch, 19712)
        #Use dropout here before fully connected for regularization
        x = self.dropout(x)
        x = self.activation_4(self.fc_layer_1(x))
        #shape: (batch, 256)
        x = self.fc_layer_2(x)
        return x

    def save_convergence_plot(self):
        #Shows learning curve
        if not self.training_history['epoch']:
            return None

        if self.plot_destination_folder_path is None:
            project_root = Path(__file__).resolve().parents[2]
            destination_dir = project_root / 'result' / 'stage_3_result'
        else:
            destination_dir = Path(self.plot_destination_folder_path)

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / self.plot_file_name

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(self.training_history['epoch'], self.training_history['loss'],
                     color='tab:red', linewidth=2, label='Train')
        axes[0].plot(self.training_history['epoch'], self.training_history['test_loss'],
                     color='tab:orange', linewidth=2, linestyle='--', label='Test')
        axes[0].set_title('CNN ORL Convergence Curve')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.training_history['epoch'], self.training_history['accuracy'],
                     color='tab:blue', linewidth=2, label='Train')
        axes[1].plot(self.training_history['epoch'], self.training_history['test_accuracy'],
                     color='tab:cyan', linewidth=2, linestyle='--', label='Test')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(destination_path, dpi=150)
        plt.close(fig)

        self.last_plot_path = str(destination_path)
        print('plot saved to:', destination_path)
        return self.last_plot_path

    def train_model(self, X, y, X_test, y_test):
        #Training loop
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        #X tensor shape (N, 1, 112, 92)
        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1) / 255.0
        #Need to subtract 1 from every label for lossfn (0 indexed)
        y_tensor = torch.LongTensor(np.array(y) - 1)

        X_test_tensor = torch.FloatTensor(np.array(X_test)).unsqueeze(1) / 255.0
        y_test_tensor = torch.LongTensor(np.array(y_test) - 1)

        N = X_tensor.shape[0]
        self.training_history = {
            'epoch': [], 'loss': [], 'accuracy': [],
            'test_loss': [], 'test_accuracy': []
        }

        for epoch in range(self.max_epoch):
            perm = torch.randperm(N) #shuffle
            X_tensor = X_tensor[perm]
            y_tensor = y_tensor[perm]

            epoch_loss = 0.0

            for start in range(0, N, self.batch_size):
                X_batch = X_tensor[start:start + self.batch_size]
                y_batch = y_tensor[start:start + self.batch_size]

                #Call self.train() mode so Dropout is active during training
                self.train()
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.shape[0]

            avg_loss = epoch_loss / N
            self.eval()
            with torch.no_grad():
                all_train_preds = self.forward(X_tensor)
                test_preds = self.forward(X_test_tensor)
                test_loss = loss_function(test_preds, y_test_tensor).item()

            accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': all_train_preds.max(1)[1]}
            train_metrics = accuracy_evaluator.evaluate()

            accuracy_evaluator.data = {'true_y': y_test_tensor, 'pred_y': test_preds.max(1)[1]}
            test_metrics = accuracy_evaluator.evaluate()

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(train_metrics['accuracy'])
            self.training_history['test_loss'].append(test_loss)
            self.training_history['test_accuracy'].append(test_metrics['accuracy'])

            if epoch % 5 == 0:
                print(f'Epoch: {epoch}  Train Acc: {train_metrics["accuracy"]:.4f}  '
                      f'Test Acc: {test_metrics["accuracy"]:.4f}  Loss: {avg_loss:.4f}')

        return self.training_history

    def test(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1) / 255.0
            y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1] + 1

    def run(self):
        print('method running--')
        print('training start...')
        history = self.train_model(
            self.data['train']['X'], self.data['train']['y'],
            self.data['test']['X'],  self.data['test']['y']
        )
        plot_path = self.save_convergence_plot()
        print('start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'training_history': history,
            'convergence_plot_path': plot_path,
        }







