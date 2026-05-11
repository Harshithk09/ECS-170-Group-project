"""
CPU-friendly CIFAR-10 classifier targeting ~87-90% test accuracy.

Architecture: Depthwise Separable CNN (MobileNet-style)
  - ~8x fewer multiply-adds than standard convolutions
  - Residual connections for gradient flow
  - Global average pooling (no giant FC layers)

Training tricks:
  - CosineAnnealingLR + linear warmup
  - Cutout augmentation
  - Label smoothing
  - AdamW optimizer

Expected: ~87-90% in ~20-30 min on CPU (50 epochs)
"""

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

_CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Architecture ──────────────────────────────────────────────────────────────

class _DSBlock(nn.Module):
    """
    Depthwise Separable block with residual connection.

    Depthwise conv (1 filter per channel) + Pointwise conv (1x1).
    ~8-9x fewer operations than a standard 3x3 conv of the same size.
    Residual added when in/out channels match.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
        self.residual = (in_ch == out_ch and stride == 1)

    def forward(self, x):
        out = self.dw(x)
        if self.residual:
            out = out + x
        return out


class EfficientCIFAR(nn.Module):
    """
    Lightweight depthwise separable CNN for 32x32 CIFAR-10.
    Channel progression: 3 -> 32 -> 64 -> 128 -> 256
    Total params: ~600k  (vs ResNet18's 11M)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = nn.Sequential(
            _DSBlock(32, 32),
            _DSBlock(32, 64, stride=2),    # 32x32 -> 16x16
        )
        self.stage2 = nn.Sequential(
            _DSBlock(64, 64),
            _DSBlock(64, 64),
            _DSBlock(64, 128, stride=2),   # 16x16 -> 8x8
        )
        self.stage3 = nn.Sequential(
            _DSBlock(128, 128),
            _DSBlock(128, 128),
            _DSBlock(128, 256, stride=2),  # 8x8 -> 4x4
        )
        self.stage4 = nn.Sequential(
            _DSBlock(256, 256),
            _DSBlock(256, 256),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(x: torch.Tensor) -> torch.Tensor:
    try:
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')
    except (NotImplementedError, RuntimeError):
        x = F.pad(x, (4, 4, 4, 4), mode='replicate')
    h = random.randint(0, 8)
    w = random.randint(0, 8)
    x = x[:, :, h:h + 32, w:w + 32]
    if random.random() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


def _cutout(x: torch.Tensor, length: int = 16) -> torch.Tensor:
    h, w   = x.shape[-2], x.shape[-1]
    cy     = random.randint(0, h - 1)
    cx     = random.randint(0, w - 1)
    y1, y2 = max(cy - length // 2, 0), min(cy + length // 2, h)
    x1, x2 = max(cx - length // 2, 0), min(cx + length // 2, w)
    mask   = torch.ones_like(x)
    mask[:, :, y1:y2, x1:x2] = 0.0
    return x * mask


# ── Method class ──────────────────────────────────────────────────────────────

class Method_CNN_CIFAR10(method):
    data                     = None
    num_classes: int         = 10
    architecture: str        = 'cnn'

    learning_rate: float     = 1e-3
    weight_decay: float      = 1e-4
    max_epoch: int           = 50        # ~20-30 min on CPU
    batch_size: int          = 256       # large batch = faster epochs on CPU
    warmup_epochs: int       = 5
    label_smoothing: float   = 0.1
    cutout_length: int       = 16
    use_augmentation: bool   = True

    plot_destination_folder_path = None
    plot_file_name: str      = 'cifar10_convergence_curve.png'
    last_plot_path           = None

    def __init__(self, mName=None, mDescription=None):
        super().__init__(mName, mDescription)
        self.device = _pick_device()
        self.model  = EfficientCIFAR(self.num_classes).to(self.device)

        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epoch - self.warmup_epochs,
            eta_min=1e-5,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.train_loss_list, self.train_acc_list = [], []
        self.test_loss_list,  self.test_acc_list  = [], []

    @staticmethod
    def _to_nchw(X_list) -> np.ndarray:
        x = np.stack([np.asarray(im, dtype=np.float32) for im in X_list])
        x /= 255.0
        if x.ndim == 4 and x.shape[-1] == 3:
            x = np.transpose(x, (0, 3, 1, 2))
        x = (x - _CIFAR_MEAN.reshape(1, 3, 1, 1)) / _CIFAR_STD.reshape(1, 3, 1, 1)
        return x

    def _loader(self, X, y, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.tensor(X),
                          torch.tensor(np.asarray(y, dtype=np.int64))),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _warmup_lr(self, epoch: int):
        lr = self.learning_rate * epoch / max(self.warmup_epochs, 1)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            if self.use_augmentation:
                xb = _augment(xb)
                xb = _cutout(xb, self.cutout_length)
            self.optimizer.zero_grad(set_to_none=True)
            out  = self.model(xb)
            loss = self.criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += xb.size(0)
        return total_loss / total, correct / total

    def _eval_epoch(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb  = xb.to(self.device)
                yb  = yb.to(self.device)
                out = self.model(xb)
                total_loss += F.cross_entropy(out, yb).item() * xb.size(0)
                correct    += (out.argmax(1) == yb).sum().item()
                total      += xb.size(0)
        return total_loss / total, correct / total

    def _predict(self, loader) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self.model(xb.to(self.device)).argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def save_convergence_plot(self):
        if not self.train_loss_list:
            return None
        if self.plot_destination_folder_path is None:
            dest = Path(__file__).resolve().parents[2] / 'result' / 'stage_3_result'
        else:
            dest = Path(self.plot_destination_folder_path)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / self.plot_file_name

        ep = range(1, len(self.train_loss_list) + 1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        a1.plot(ep, self.train_loss_list, label='Train', color='steelblue')
        a1.plot(ep, self.test_loss_list,  label='Test',  color='tomato', linestyle='--')
        a1.set_title('Loss'); a1.set_xlabel('Epoch'); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(ep, self.train_acc_list,  label='Train', color='steelblue')
        a2.plot(ep, self.test_acc_list,   label='Test',  color='tomato', linestyle='--')
        a2.set_title('Accuracy'); a2.set_xlabel('Epoch'); a2.legend(); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        self.last_plot_path = str(path)
        print('convergence plot saved to:', path)
        return self.last_plot_path

    def run(self):
        if self.data is None:
            raise ValueError('method.data must be set before run().')

        print(f'device: {self.device}  |  epochs: {self.max_epoch}  '
              f'|  batch: {self.batch_size}')

        X_train = self._to_nchw(self.data['train']['X'])
        y_train = np.asarray(self.data['train']['y'])
        X_test  = self._to_nchw(self.data['test']['X'])
        y_test  = np.asarray(self.data['test']['y'])

        train_loader = self._loader(X_train, y_train, shuffle=True)
        test_loader  = self._loader(X_test,  y_test,  shuffle=False)

        print('--start training...')
        for epoch in range(1, self.max_epoch + 1):
            if epoch <= self.warmup_epochs:
                self._warmup_lr(epoch)

            tr_loss, tr_acc = self._train_epoch(train_loader)
            te_loss, te_acc = self._eval_epoch(test_loader)

            if epoch > self.warmup_epochs:
                self.scheduler.step()

            self.train_loss_list.append(tr_loss)
            self.train_acc_list.append(tr_acc)
            self.test_loss_list.append(te_loss)
            self.test_acc_list.append(te_acc)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch:>2}/{self.max_epoch}]  LR: {lr_now:.5f}  '
                f'Train: {tr_loss:.4f}/{tr_acc:.4f}  '
                f'Test:  {te_loss:.4f}/{te_acc:.4f}'
            )

        plot_path = self.save_convergence_plot()
        print('--start testing...')
        pred_y = self._predict(test_loader)

        return {
            'pred_y': pred_y,
            'true_y': y_test,
            'training_history': {
                'epoch':          list(range(1, self.max_epoch + 1)),
                'train_loss':     self.train_loss_list,
                'test_loss':      self.test_loss_list,
                'train_accuracy': self.train_acc_list,
                'test_accuracy':  self.test_acc_list,
            },
            'convergence_plot_path': plot_path,
        }