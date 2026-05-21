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
from torch.utils.data import DataLoader, TensorDataset


class Method_RNN_Classification(method, nn.Module):
    data = None

    # ── Hyperparameters ───────────────────────────────────────
    # Vocab size will be set after the dataset loader builds the vocab
    vocab_size   = 20000
    embed_dim    = 64        # size of each word embedding vector
    hidden_dim   = 128       # number of hidden units in the RNN
    num_layers   = 2         # number of stacked RNN layers
    num_classes  = 2         # pos / neg
    max_epoch    = 10        # epochs to train
    learning_rate = 1e-3
    batch_size   = 64
    dropout      = 0.3

    # cell_type controls which recurrent unit is used:
    #   'rnn'  → vanilla RNN  (baseline, prone to vanishing gradients)
    #   'lstm' → Long Short-Term Memory (handles long sequences better)
    #   'gru'  → Gated Recurrent Unit (lighter than LSTM, similar performance)
    cell_type = 'lstm'

    # Plot settings
    plot_destination_folder_path = None
    plot_file_name = 'RNN_classification_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # ── Embedding layer ───────────────────────────────────
        # Converts integer token IDs into dense vectors of size embed_dim.
        # padding_idx=0 tells PyTorch to always output a zero vector for
        # the <PAD> token and never update its weights.
        self.embedding = nn.Embedding(
            self.vocab_size, self.embed_dim, padding_idx=0)

        # ── Recurrent layer ───────────────────────────────────
        # Picks the right class based on cell_type so we can swap
        # RNN / LSTM / GRU just by changing one attribute.
        rnn_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        rnn_cls = rnn_map[self.cell_type.lower()]
        self.rnn = rnn_cls(
            input_size    = self.embed_dim,
            hidden_size   = self.hidden_dim,
            num_layers    = self.num_layers,
            # dropout only applied between layers, not after the last one
            dropout       = self.dropout if self.num_layers > 1 else 0.0,
            batch_first   = True,    # input shape: (batch, seq_len, features)
            bidirectional = True,    # reads sequence left→right AND right→left
        )

        self.drop = nn.Dropout(self.dropout)

        # ── Output layer ──────────────────────────────────────
        # Bidirectional doubles the hidden size, so input is hidden_dim * 2
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

        # Training history for convergence plot
        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}

        # Auto-select device: CUDA GPU → Apple MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')

    def forward(self, x):
        '''
        Forward pass through embedding → RNN → FC.

        x shape: (batch_size, seq_len)  — integer token IDs
        '''
        # 1. Embed tokens: (batch, seq_len) → (batch, seq_len, embed_dim)
        emb = self.drop(self.embedding(x))

        # 2. Pass through RNN
        #    out shape  : (batch, seq_len, hidden_dim * 2)  [bidirectional]
        #    hidden shape: (num_layers * 2, batch, hidden_dim)
        if self.cell_type.lower() == 'lstm':
            out, (hidden, cell) = self.rnn(emb)
        else:
            out, hidden = self.rnn(emb)

        # 3. Grab the last layer's forward and backward hidden states
        #    hidden[-2] = last forward layer, hidden[-1] = last backward layer
        fwd = hidden[-2]                            # (batch, hidden_dim)
        bwd = hidden[-1]                            # (batch, hidden_dim)
        combined = torch.cat([fwd, bwd], dim=1)     # (batch, hidden_dim * 2)

        # 4. Classify
        logits = self.fc(self.drop(combined))       # (batch, num_classes)
        return logits

    def save_convergence_plot(self):
        if not self.training_history['epoch']:
            return None

        if self.plot_destination_folder_path is None:
            project_root = Path(__file__).resolve().parents[2]
            destination_dir = project_root / 'result' / 'stage_4_result'
        else:
            destination_dir = Path(self.plot_destination_folder_path)

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / self.plot_file_name

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(self.training_history['epoch'],
                     self.training_history['loss'],
                     color='tab:red', linewidth=2)
        axes[0].set_title(f'RNN Classification Convergence ({self.cell_type.upper()})')
        axes[0].set_ylabel('Training Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.training_history['epoch'],
                     self.training_history['accuracy'],
                     color='tab:blue', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Training Accuracy')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(destination_path, dpi=150)
        plt.close(fig)

        self.last_plot_path = str(destination_path)
        print('Convergence plot saved to:', destination_path)
        return self.last_plot_path

    def fit(self, train_loader):
        '''
        Train the model using mini-batches from the DataLoader.
        '''
        self.to(self.device)
        optimizer     = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}

        for epoch in range(self.max_epoch):
            self.train_mode()
            total_loss, correct, total = 0.0, 0, 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                y_pred = self.forward(xb)
                loss   = loss_function(y_pred, yb)

                optimizer.zero_grad()
                loss.backward()
                # Clip gradients — critical for RNN to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(yb)
                correct    += (y_pred.max(1)[1] == yb).sum().item()
                total      += len(yb)

            avg_loss = total_loss / total
            accuracy = correct / total

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)

            if epoch % 2 == 0:
                print(f'Epoch: {epoch+1}  '
                      f'Loss: {avg_loss:.4f}  '
                      f'Accuracy: {accuracy:.4f}')

        return self.training_history

    def test(self, test_loader):
        '''Run inference on the test set and return predictions + true labels.'''
        self.eval_mode()
        all_preds, all_true = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                preds = self.forward(xb).max(1)[1].cpu()
                all_preds.extend(preds.tolist())
                all_true.extend(yb.tolist())

        return all_preds, all_true

    def train_mode(self):
        '''Set model to training mode (enables dropout).'''
        super(nn.Module, self).__setattr__('training', True)
        nn.Module.train(self)

    def eval_mode(self):
        '''Set model to eval mode (disables dropout).'''
        nn.Module.eval(self)

    def run(self):
        print('method running...')
        print('--start training...')
        history = self.fit(self.data['train_loader'])
        plot_path = self.save_convergence_plot()
        print('--start testing...')
        pred_y, true_y = self.test(self.data['test_loader'])
        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'training_history': history,
            'convergence_plot_path': plot_path,
        }