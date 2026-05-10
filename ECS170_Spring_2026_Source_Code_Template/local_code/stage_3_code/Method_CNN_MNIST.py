'''
Concrete CNN MethodModule class for MNIST image classification
'''

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 30
    learning_rate = 1e-3
    training_history = None
    plot_destination_folder_path = None
    plot_file_name = 'CNN_MNIST_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv_layer_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.activation_func_1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_1 = nn.Linear(64 * 5 * 5, 128)
        self.activation_func_3 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, 10)

        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}

    def forward(self, x):
        x = self.pool(self.activation_func_1(self.conv_layer_1(x)))
        x = self.pool(self.activation_func_2(self.conv_layer_2(x)))

        x = x.view(x.size(0), -1)

        x = self.activation_func_3(self.fc_layer_1(x))
        y_pred = self.fc_layer_2(x)

        return y_pred

    def prepare_images(self, X):
        X = np.array(X)
        X = torch.FloatTensor(X)

        # MNIST images need shape: batch, channel, height, width
        if len(X.shape) == 3:
            X = X.unsqueeze(1)

        return X

    def save_convergence_plot(self):
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

        axes[0].plot(self.training_history['epoch'], self.training_history['loss'], linewidth=2)
        axes[0].set_title('CNN MNIST Convergence Curve')
        axes[0].set_ylabel('Training Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.training_history['epoch'], self.training_history['accuracy'], linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Training Accuracy')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(destination_path, dpi=150)
        plt.close(fig)

        self.last_plot_path = str(destination_path)
        print('convergence plot saved to:', destination_path)
        return self.last_plot_path

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}

        X_tensor = self.prepare_images(X)
        y_true = torch.LongTensor(np.array(y))

        batch_size = 64

        for epoch in range(self.max_epoch):
            permutation = torch.randperm(X_tensor.size(0))
            total_loss = 0
            all_pred = []
            all_true = []

            for i in range(0, X_tensor.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X = X_tensor[indices]
                batch_y = y_true[indices]

                y_pred = self.forward(batch_X)
                train_loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item()
                all_pred.extend(y_pred.max(1)[1].tolist())
                all_true.extend(batch_y.tolist())

            accuracy_evaluator.data = {
                'true_y': torch.LongTensor(all_true),
                'pred_y': torch.LongTensor(all_pred)
            }

            train_metrics = accuracy_evaluator.evaluate()
            avg_loss = total_loss / (X_tensor.size(0) / batch_size)

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(train_metrics['accuracy'])

            print('Epoch:', epoch + 1, 'Accuracy:', train_metrics['accuracy'], 'Loss:', avg_loss)

        return self.training_history

    def test(self, X):
        X_tensor = self.prepare_images(X)

        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        history = self.train(self.data['train']['X'], self.data['train']['y'])

        plot_path = self.save_convergence_plot()

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'training_history': history,
            'convergence_plot_path': plot_path,
        }