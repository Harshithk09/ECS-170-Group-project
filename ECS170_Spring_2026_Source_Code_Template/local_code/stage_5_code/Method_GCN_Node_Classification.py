'''
GCN Method for Node Classification (Stage 5)

Implements a 2-layer Graph Convolutional Network following:
  Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017.

Each GCN layer computes:
  H^(l+1) = sigma( A_hat * H^(l) * W^(l) )
where A_hat = D^{-1/2} (A + I) D^{-1/2}  (pre-computed in the dataset loader)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from code.base_class.method import method


# ---------------------------------------------------------------------------
# Single GCN layer: H_out = A_hat * H_in * W  (no bias standard variant)
# ---------------------------------------------------------------------------
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._init_weights()

    def _init_weights(self):
        # Glorot (Xavier) uniform initialisation
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        # support = X * W
        support = torch.mm(x, self.weight)
        # output = A_hat * support  (sparse-dense multiplication)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return (self.__class__.__name__ + '('
                + str(self.in_features) + ' -> '
                + str(self.out_features) + ')')


# ---------------------------------------------------------------------------
# 2-layer GCN
# ---------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvLayer(n_features, n_hidden)
        self.gc2 = GraphConvLayer(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        # Layer 1: ReLU + dropout
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2: raw logits (CrossEntropyLoss includes log-softmax)
        x = self.gc2(x, adj)
        return x

    def embed(self, x, adj):
        """Return the hidden-layer (pre-dropout) node embeddings."""
        with torch.no_grad():
            emb = F.relu(self.gc1(x, adj))
        return emb


# ---------------------------------------------------------------------------
# Training / evaluation wrapper that fits into the course template
# ---------------------------------------------------------------------------
class Method_GCN_Node_Classification(method):

    # ---- hyper-parameters (can be overridden before calling run()) --------
    learning_rate = 0.01
    weight_decay  = 5e-4
    n_hidden      = 64
    dropout       = 0.5
    max_epoch     = 200

    # set from outside
    data          = None
    dataset_name  = 'unknown'
    result_folder = 'result'

    def __init__(self, mName=None, mDescription=None):
        super(Method_GCN_Node_Classification, self).__init__(mName, mDescription)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def accuracy(logits, labels, idx):
        preds = logits[idx].max(1)[1]
        correct = preds.eq(labels[idx]).sum().item()
        return correct / len(idx)

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train(self):
        # ---- unpack data ----
        adj    = self.data['utility']['A']
        X      = self.data['X']
        y      = self.data['y']
        idx_tr = self.data['idx_train']
        idx_va = self.data['idx_val']

        n_features = X.shape[1]
        n_classes  = int(y.max().item()) + 1

        # ---- move to device ----
        device = self.device
        adj, X, y = adj.to(device), X.to(device), y.to(device)
        idx_tr = idx_tr.to(device)
        idx_va = idx_va.to(device)

        # ---- build model ----
        model = GCN(n_features, self.n_hidden, n_classes, self.dropout).to(device)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # ---- history buffers ----
        train_loss_hist, val_loss_hist = [], []
        train_acc_hist,  val_acc_hist  = [], []

        print(f'\n{"="*60}')
        print(f'  Dataset : {self.dataset_name}')
        print(f'  Model   : {model}')
        print(f'  Device  : {device}')
        print(f'  Classes : {n_classes}  |  Features : {n_features}')
        print(f'{"="*60}')

        t0 = time.time()
        for epoch in range(1, self.max_epoch + 1):
            # -- train step --
            model.train()
            optimizer.zero_grad()
            logits = model(X, adj)
            loss_tr = criterion(logits[idx_tr], y[idx_tr])
            loss_tr.backward()
            optimizer.step()

            # -- validation step --
            model.eval()
            with torch.no_grad():
                logits = model(X, adj)
                loss_va = criterion(logits[idx_va], y[idx_va]).item()

            acc_tr = self.accuracy(logits, y, idx_tr)
            acc_va = self.accuracy(logits, y, idx_va)

            train_loss_hist.append(loss_tr.item())
            val_loss_hist.append(loss_va)
            train_acc_hist.append(acc_tr)
            val_acc_hist.append(acc_va)

            if epoch % 20 == 0 or epoch == 1:
                print(f'  Epoch {epoch:>3d} | '
                      f'Train Loss {loss_tr.item():.4f} Acc {acc_tr:.4f} | '
                      f'Val Loss {loss_va:.4f} Acc {acc_va:.4f}')

        elapsed = time.time() - t0
        print(f'\n  Training time: {elapsed:.1f}s')

        return model, {
            'train_loss': train_loss_hist,
            'val_loss':   val_loss_hist,
            'train_acc':  train_acc_hist,
            'val_acc':    val_acc_hist,
        }

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------
    def test(self, model):
        adj   = self.data['utility']['A'].to(self.device)
        X     = self.data['X'].to(self.device)
        y     = self.data['y'].to(self.device)
        idx_te = self.data['idx_test'].to(self.device)

        model.eval()
        with torch.no_grad():
            logits = model(X, adj)

        preds = logits[idx_te].max(1)[1].cpu()
        true  = y[idx_te].cpu()
        acc   = (preds == true).float().mean().item()
        return preds, true, acc

    # ------------------------------------------------------------------
    # plot learning curves
    # ------------------------------------------------------------------
    def plot_curves(self, history):
        os.makedirs(self.result_folder, exist_ok=True)
        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Learning Curves – {self.dataset_name}', fontsize=14, fontweight='bold')

        # Loss
        axes[0].plot(epochs, history['train_loss'], label='Train Loss',   color='steelblue')
        axes[0].plot(epochs, history['val_loss'],   label='Val Loss',     color='coral',  linestyle='--')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, history['train_acc'], label='Train Accuracy', color='steelblue')
        axes[1].plot(epochs, history['val_acc'],   label='Val Accuracy',   color='coral',  linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.result_folder,
                                 f'learning_curve_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Curve saved → {save_path}')
        return save_path

    # ------------------------------------------------------------------
    # run()  – called by the Setting
    # ------------------------------------------------------------------
    def run(self):
        # detect device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model, history = self.train()
        preds, true, test_acc = self.test(model)
        curve_path = self.plot_curves(history)

        print(f'\n  *** Test Accuracy ({self.dataset_name}): {test_acc:.4f} ***\n')

        return {
            'pred_y':    preds,
            'true_y':    true,
            'test_acc':  test_acc,
            'history':   history,
            'curve':     curve_path,
            'model':     model,
        }
