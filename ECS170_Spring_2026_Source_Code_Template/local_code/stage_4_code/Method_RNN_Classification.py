'''
Concrete MethodModule class for RNN-based Text Classification
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


class Method_RNN_Classification(method, nn.Module):
    data = None

    # ── Hyperparameters ───────────────────────────────────────
    vocab_size    = 20000
    embed_dim     = 64
    hidden_dim    = 128
    num_layers    = 2
    num_classes   = 2
    max_epoch     = 10
    learning_rate = 1e-3
    batch_size    = 64
    dropout       = 0.3

    # cell_type: 'rnn' | 'lstm' | 'gru'
    cell_type = 'lstm'

    plot_destination_folder_path = None
    plot_file_name = 'RNN_classification_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Embedding layer — maps token IDs to dense vectors
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Recurrent layer — swap cell type via self.cell_type
        rnn_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        rnn_cls = rnn_map[self.cell_type.lower()]
        self.rnn = rnn_cls(
            input_size    = self.embed_dim,
            hidden_size   = self.hidden_dim,
            num_layers    = self.num_layers,
            dropout       = self.dropout if self.num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = True,   # reads sequence both directions
        )

        self.drop = nn.Dropout(self.dropout)
        # bidirectional doubles hidden size
        self.fc   = nn.Linear(self.hidden_dim * 2, self.num_classes)

        self.training_history = {
            'epoch': [],
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
        }

        # Auto-select device: CUDA → MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')

    def forward(self, x):
        # x: (batch, seq_len) integer token IDs
        emb = self.drop(self.embedding(x))              # (batch, seq_len, embed_dim)

        if self.cell_type.lower() == 'lstm':
            out, (hidden, cell) = self.rnn(emb)
        else:
            out, hidden = self.rnn(emb)

        # Concat last forward + backward hidden states
        fwd      = hidden[-2]                           # (batch, hidden_dim)
        bwd      = hidden[-1]                           # (batch, hidden_dim)
        combined = torch.cat([fwd, bwd], dim=1)         # (batch, hidden_dim*2)
        logits   = self.fc(self.drop(combined))         # (batch, num_classes)
        return logits

    def save_convergence_plot(self):
        if not self.training_history['epoch']:
            return None

        if self.plot_destination_folder_path is None:
            project_root    = Path(__file__).resolve().parents[2]
            destination_dir = project_root / 'result' / 'stage_4_result'
        else:
            destination_dir = Path(self.plot_destination_folder_path)

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / self.plot_file_name

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # ── Loss curve ────────────────────────────────────────
        axes[0].plot(self.training_history['epoch'],
                     self.training_history['train_loss'],
                     color='tab:red',    linewidth=2, label='Train')
        axes[0].plot(self.training_history['epoch'],
                     self.training_history['val_loss'],
                     color='tab:orange', linewidth=2, label='Val', linestyle='--')
        axes[0].set_title(f'RNN Classification Convergence ({self.cell_type.upper()})')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ── Accuracy curve ────────────────────────────────────
        axes[1].plot(self.training_history['epoch'],
                     self.training_history['train_acc'],
                     color='tab:blue', linewidth=2, label='Train')
        axes[1].plot(self.training_history['epoch'],
                     self.training_history['val_acc'],
                     color='tab:cyan', linewidth=2, label='Val', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(destination_path, dpi=150)
        plt.close(fig)

        self.last_plot_path = str(destination_path)
        print('Convergence plot saved to:', destination_path)
        return self.last_plot_path

    def fit(self, train_loader, val_loader=None):
        '''Train the model with mini-batch SGD, tracking train and val curves.'''
        self.to(self.device)
        optimizer     = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        self.training_history = {
            'epoch': [],
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
        }

        for epoch in range(self.max_epoch):

            # ── Training pass ──────────────────────────────────
            nn.Module.train(self)
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                y_pred = self.forward(xb)
                loss   = loss_function(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (y_pred.max(1)[1] == yb).sum().item()
                total      += len(yb)

            train_loss = total_loss / total
            train_acc  = correct / total

            # ── Validation pass ────────────────────────────────
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                nn.Module.eval(self)
                v_loss, v_correct, v_total = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        y_pred = self.forward(xb)
                        loss   = loss_function(y_pred, yb)
                        v_loss    += loss.item() * len(yb)
                        v_correct += (y_pred.max(1)[1] == yb).sum().item()
                        v_total   += len(yb)
                val_loss = v_loss / v_total
                val_acc  = v_correct / v_total

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            if epoch % 2 == 0:
                print(f'Epoch: {epoch+1}  '
                      f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  '
                      f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}')

        return self.training_history

    def test(self, test_loader):
        '''Run inference on test set, return (predictions, true labels).'''
        nn.Module.eval(self)
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb    = xb.to(self.device)
                preds = self.forward(xb).max(1)[1].cpu()
                all_preds.extend(preds.tolist())
                all_true.extend(yb.tolist())
        return all_preds, all_true

    def run(self):
        print('method running...')
        print('--start training...')
        history = self.fit(
            self.data['train_loader'],
            self.data.get('val_loader'),
        )
        plot_path = self.save_convergence_plot()
        print('--start testing...')
        pred_y, true_y = self.test(self.data['test_loader'])
        return {
            'pred_y':               pred_y,
            'true_y':               true_y,
            'training_history':     history,
            'convergence_plot_path': plot_path,
        }