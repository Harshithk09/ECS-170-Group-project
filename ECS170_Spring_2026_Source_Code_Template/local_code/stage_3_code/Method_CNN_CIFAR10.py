'''
CNN / ResNet-18 classifier for CIFAR-10 (32x32x3). Default `architecture='resnet18'`
uses the standard CIFAR setup (normalize, flip + pad/crop aug, SGD, MultiStep LR)
to reach strong test accuracy; set `architecture='cnn'` for the smaller baseline CNN.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


def _pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _resnet18_cifar(num_classes=10):
    from torchvision.models import resnet18

    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class CNN_Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class Method_CNN_CIFAR10(method):
    data = None
    num_classes = 10
    architecture = 'resnet18'
    learning_rate = 1e-3
    resnet_lr = 0.1
    max_epoch = 200
    batch_size = 128
    use_augmentation = True
    use_cifar_normalize = True
    plot_destination_folder_path = None
    plot_file_name = 'cifar10_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName=None, mDescription=None):
        super().__init__(mName, mDescription)
        self.device = _pick_device()
        if self.architecture == 'resnet18':
            self.model = _resnet18_cifar(self.num_classes).to(self.device)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.resnet_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[100, 150], gamma=0.1
            )
        else:
            self.model = CNN_Net(num_classes=self.num_classes).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []

    def _normalize_cifar(self, x_nchw):
        if not self.use_cifar_normalize:
            return x_nchw
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x_nchw.device, dtype=x_nchw.dtype)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=x_nchw.device, dtype=x_nchw.dtype)
        return (x_nchw - mean[None, :, None, None]) / std[None, :, None, None]

    def _stack_inputs(self, X_list):
        x = np.stack([np.asarray(im, dtype=np.float32) for im in X_list], axis=0)
        x = x / 255.0
        if x.ndim == 4 and x.shape[-1] == 3:
            x = np.transpose(x, (0, 3, 1, 2))
        if self.use_cifar_normalize:
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
            x = (x - mean) / std
        return x

    def _augment_batch(self, x):
        if not self.use_augmentation or self.architecture != 'resnet18':
            return x
        try:
            x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        except (NotImplementedError, RuntimeError):
            x = F.pad(x, (4, 4, 4, 4), mode='replicate')
        h_off = random.randint(0, 8)
        w_off = random.randint(0, 8)
        x = x[:, :, h_off : h_off + 32, w_off : w_off + 32]
        if random.random() < 0.5:
            x = torch.flip(x, dims=(3,))
        return x

    def _make_loader(self, X_arr, y_arr, shuffle):
        xs = torch.tensor(X_arr)
        ys = torch.tensor(np.asarray(y_arr, dtype=np.int64))
        return DataLoader(TensorDataset(xs, ys), batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            if self.architecture == 'resnet18' and self.use_augmentation:
                X_batch = self._normalize_cifar(X_batch)
                X_batch = self._augment_batch(X_batch)
            elif self.architecture == 'resnet18':
                X_batch = self._normalize_cifar(X_batch)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
        return total_loss / total, correct / total

    def _eval_epoch(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                if self.architecture == 'resnet18':
                    X_batch = self._normalize_cifar(X_batch)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)
        return total_loss / total, correct / total

    def save_convergence_plot(self):
        if not self.train_loss_list:
            return None
        if self.plot_destination_folder_path is None:
            project_root = Path(__file__).resolve().parents[2]
            destination_dir = project_root / 'result' / 'stage_3_result'
        else:
            destination_dir = Path(self.plot_destination_folder_path)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / self.plot_file_name

        epochs = range(1, len(self.train_loss_list) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, self.train_loss_list, label='Train Loss', color='steelblue')
        ax1.plot(epochs, self.test_loss_list, label='Test Loss', color='tomato', linestyle='--')
        ax1.set_title('Loss (CIFAR-10)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.train_acc_list, label='Train Accuracy', color='steelblue')
        ax2.plot(epochs, self.test_acc_list, label='Test Accuracy', color='tomato', linestyle='--')
        ax2.set_title('Accuracy (CIFAR-10)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(destination_path, dpi=150)
        plt.close(fig)
        self.last_plot_path = str(destination_path)
        print('convergence plot saved to:', destination_path)
        return self.last_plot_path

    def _predict_loader(self, loader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                if self.architecture == 'resnet18':
                    X_batch = self._normalize_cifar(X_batch)
                outputs = self.model(X_batch)
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    def run(self):
        if self.data is None:
            raise ValueError('method.data must be set before run().')

        print('method running...')
        print('device:', self.device)
        print('architecture:', self.architecture)

        if self.architecture == 'resnet18':
            X_train = np.stack([np.asarray(im, dtype=np.float32) for im in self.data['train']['X']], axis=0)
            X_train = X_train / 255.0
            if X_train.ndim == 4 and X_train.shape[-1] == 3:
                X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.stack([np.asarray(im, dtype=np.float32) for im in self.data['test']['X']], axis=0)
            X_test = X_test / 255.0
            if X_test.ndim == 4 and X_test.shape[-1] == 3:
                X_test = np.transpose(X_test, (0, 3, 1, 2))
            y_train = np.asarray(self.data['train']['y'])
            y_test = np.asarray(self.data['test']['y'])
        else:
            X_train = self._stack_inputs(self.data['train']['X'])
            y_train = np.asarray(self.data['train']['y'])
            X_test = self._stack_inputs(self.data['test']['X'])
            y_test = np.asarray(self.data['test']['y'])

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        test_loader = self._make_loader(X_test, y_test, shuffle=False)

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []

        print('--start training...')
        for epoch in range(1, self.max_epoch + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            test_loss, test_acc = self._eval_epoch(test_loader)
            self.scheduler.step()

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.test_loss_list.append(test_loss)
            self.test_acc_list.append(test_acc)

            if epoch == 1 or epoch % 10 == 0 or epoch == self.max_epoch:
                print(
                    f'Epoch [{epoch:>3}/{self.max_epoch}] '
                    f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  '
                    f'Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}'
                )

        plot_path = self.save_convergence_plot()
        print('--start testing...')
        pred_y = self._predict_loader(test_loader)

        history = {
            'epoch': list(range(1, self.max_epoch + 1)),
            'train_loss': self.train_loss_list,
            'test_loss': self.test_loss_list,
            'train_accuracy': self.train_acc_list,
            'test_accuracy': self.test_acc_list,
        }
        return {
            'pred_y': pred_y,
            'true_y': y_test,
            'training_history': history,
            'convergence_plot_path': plot_path,
        }
