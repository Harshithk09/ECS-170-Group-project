'''
Concrete MethodModule class for RNN-based Text Generation
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import math


class Method_RNN_Generation(method, nn.Module):
    data = None

    # ── Hyperparameters ───────────────────────────────────────
    vocab_size    = 8000
    embed_dim     = 128
    hidden_dim    = 256
    num_layers    = 2
    max_epoch     = 20
    learning_rate = 2e-3
    batch_size    = 64
    dropout       = 0.3

    # cell_type: 'rnn' | 'lstm' | 'gru'
    cell_type = 'lstm'

    # Generation settings
    gen_length  = 50
    temperature = 0.8

    plot_destination_folder_path = None
    plot_file_name = 'RNN_generation_convergence_curve.png'
    last_plot_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        rnn_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        rnn_cls = rnn_map[self.cell_type.lower()]
        self.rnn = rnn_cls(
            input_size  = self.embed_dim,
            hidden_size = self.hidden_dim,
            num_layers  = self.num_layers,
            dropout     = self.dropout if self.num_layers > 1 else 0.0,
            batch_first = True,
            # Unidirectional — can only use past context when generating
        )

        self.drop = nn.Dropout(self.dropout)
        self.fc   = nn.Linear(self.hidden_dim, self.vocab_size)

        self.training_history = {
            'epoch': [],
            'train_loss': [], 'train_ppl': [],
        }

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        emb            = self.drop(self.embedding(x))
        out, hidden    = self.rnn(emb, hidden)
        logits         = self.fc(self.drop(out))    # (batch, seq_len, vocab_size)
        return logits, hidden

    def detach_hidden(self, hidden):
        if self.cell_type.lower() == 'lstm':
            return tuple(h.detach() for h in hidden)
        return hidden.detach()

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
                     color='tab:red', linewidth=2, label='Train')
        axes[0].set_title(f'RNN Generation Convergence ({self.cell_type.upper()})')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ── Perplexity curve ──────────────────────────────────
        axes[1].plot(self.training_history['epoch'],
                     self.training_history['train_ppl'],
                     color='tab:green', linewidth=2, label='Train')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Perplexity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(destination_path, dpi=150)
        plt.close(fig)

        self.last_plot_path = str(destination_path)
        print('Convergence plot saved to:', destination_path)
        return self.last_plot_path

    def fit(self, train_loader):
        '''Train the language model using next-word prediction.'''
        self.to(self.device)
        optimizer     = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)

        self.training_history = {'epoch': [], 'train_loss': [], 'train_ppl': []}
        hidden = None

        for epoch in range(self.max_epoch):
            nn.Module.train(self)
            total_loss, total_tokens = 0.0, 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                if hidden is not None:
                    hidden = self.detach_hidden(hidden)

                logits, hidden = self.forward(xb, hidden)
                loss = loss_function(
                    logits.reshape(-1, self.vocab_size),
                    yb.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss   += loss.item() * yb.numel()
                total_tokens += yb.numel()

            avg_loss   = total_loss / total_tokens
            perplexity = math.exp(min(avg_loss, 100))
            scheduler.step(avg_loss)

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_ppl'].append(perplexity)

            if epoch % 2 == 0:
                print(f'Epoch: {epoch+1}  '
                      f'Loss: {avg_loss:.4f}  '
                      f'Perplexity: {perplexity:.2f}')

        return self.training_history

    def generate(self, start_words, vocab, id2word):
        '''Generate text from a list of seed words.'''
        nn.Module.eval(self)
        self.to(self.device)

        token_ids = [vocab.get(w.lower(), 1) for w in start_words]
        generated = list(start_words)
        hidden    = None

        # Warm up hidden state on seed
        seed_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, hidden = self.forward(seed_tensor, hidden)

        current = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(self.gen_length):
                logits, hidden = self.forward(current, hidden)
                logits  = logits[:, -1, :] / self.temperature
                probs   = F.softmax(logits, dim=-1).squeeze(0)
                next_id = torch.multinomial(probs, 1).item()
                generated.append(id2word.get(next_id, '<UNK>'))
                current = torch.tensor([[next_id]], dtype=torch.long, device=self.device)

        return ' '.join(generated)

    def run(self):
        print('method running...')
        print('--start training...')
        history = self.fit(self.data['train_loader'])
        plot_path = self.save_convergence_plot()
        print('--start generating...')
        generated_text = self.generate(
            self.data['start_words'],
            self.data['vocab'],
            self.data['id2word'],
        )
        print('Generated text:')
        print(generated_text)
        return {
            'generated_text':        generated_text,
            'training_history':      history,
            'convergence_plot_path': plot_path,
        }