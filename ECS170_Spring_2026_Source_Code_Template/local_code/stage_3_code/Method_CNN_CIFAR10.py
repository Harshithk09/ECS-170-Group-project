'''
High-accuracy CIFAR-10 classifier.

The default architecture is WideResNet-28-10 with CIFAR-style augmentation,
SGD + Nesterov momentum, cosine learning-rate decay, label smoothing, Cutout,
and optional MixUp. Set ``architecture='resnet18'`` or ``architecture='cnn'``
for lighter baselines.
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


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = _conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = _conv3x3(out_planes, out_planes, 1)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError('WideResNet depth must be 6n + 4.')
        blocks_per_group = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.in_planes = widths[0]

        self.conv1 = _conv3x3(3, widths[0])
        self.block1 = self._make_group(widths[1], blocks_per_group, dropout_rate, stride=1)
        self.block2 = self._make_group(widths[2], blocks_per_group, dropout_rate, stride=2)
        self.block3 = self._make_group(widths[3], blocks_per_group, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(widths[3], num_classes)
        self._initialize()

    def _make_group(self, out_planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(WideBasicBlock(self.in_planes, out_planes, dropout_rate, block_stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


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
    architecture = 'wideresnet'
    learning_rate = 1e-3
    cifar_lr = 0.1
    max_epoch = 300
    batch_size = 128
    use_augmentation = True
    use_cifar_normalize = True
    use_cutout = True
    cutout_size = 16
    use_mixup = True
    mixup_alpha = 1.0
    label_smoothing = 0.1
    wrn_depth = 28
    wrn_widen_factor = 10
    wrn_dropout_rate = 0.3
    plot_destination_folder_path = None
    plot_file_name = 'cifar10_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName=None, mDescription=None):
        super().__init__(mName, mDescription)
        self.device = _pick_device()
        if self.architecture == 'wideresnet':
            self.model = WideResNet(
                depth=self.wrn_depth,
                widen_factor=self.wrn_widen_factor,
                dropout_rate=self.wrn_dropout_rate,
                num_classes=self.num_classes,
            ).to(self.device)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cifar_lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epoch,
                eta_min=1e-5,
            )
        elif self.architecture == 'resnet18':
            self.model = _resnet18_cifar(self.num_classes).to(self.device)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cifar_lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epoch,
                eta_min=1e-5,
            )
        else:
            self.model = CNN_Net(num_classes=self.num_classes).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

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
        return x

    def _augment_batch(self, x):
        if not self.use_augmentation or self.architecture not in {'wideresnet', 'resnet18'}:
            return x
        try:
            x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        except (NotImplementedError, RuntimeError):
            x = F.pad(x, (4, 4, 4, 4), mode='replicate')
        cropped = []
        for img in x:
            h_off = random.randint(0, 8)
            w_off = random.randint(0, 8)
            img = img[:, h_off:h_off + 32, w_off:w_off + 32]
            if random.random() < 0.5:
                img = torch.flip(img, dims=(2,))
            cropped.append(img)
        x = torch.stack(cropped)
        if self.use_cutout:
            x = self._cutout(x)
        return x

    def _cutout(self, x):
        if self.cutout_size <= 0:
            return x
        _, _, height, width = x.shape
        half = self.cutout_size // 2
        for img in x:
            center_y = random.randint(0, height - 1)
            center_x = random.randint(0, width - 1)
            y1 = max(0, center_y - half)
            y2 = min(height, center_y + half)
            x1 = max(0, center_x - half)
            x2 = min(width, center_x + half)
            img[:, y1:y2, x1:x2] = 0.0
        return x

    def _mixup(self, x, y):
        if not self.use_mixup or self.mixup_alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[indices]
        return mixed_x, y, y[indices], lam

    def _classification_loss(self, outputs, y_a, y_b=None, lam=1.0):
        if y_b is None or lam == 1.0:
            return self.criterion(outputs, y_a)
        return lam * self.criterion(outputs, y_a) + (1.0 - lam) * self.criterion(outputs, y_b)

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
            if self.architecture in {'wideresnet', 'resnet18'} and self.use_augmentation:
                X_batch = self._normalize_cifar(X_batch)
                X_batch = self._augment_batch(X_batch)
            elif self.architecture in {'wideresnet', 'resnet18'}:
                X_batch = self._normalize_cifar(X_batch)
            X_batch, y_a, y_b, lam = self._mixup(X_batch, y_batch)
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(X_batch)
                loss = self._classification_loss(outputs, y_a, y_b, lam)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
                if self.architecture in {'wideresnet', 'resnet18'}:
                    X_batch = self._normalize_cifar(X_batch)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
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
                if self.architecture in {'wideresnet', 'resnet18'}:
                    X_batch = self._normalize_cifar(X_batch)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(X_batch)
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    def run(self):
        if self.data is None:
            raise ValueError('method.data must be set before run().')

        print('method running...')
        print('device:', self.device)
        print('architecture:', self.architecture)

        if self.architecture in {'wideresnet', 'resnet18'}:
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
