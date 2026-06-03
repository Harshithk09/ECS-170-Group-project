'''
Concrete method class for GCN-based node classification (Stage 5)

Inherits from both:
  - method (course base class from code.base_class.method)
  - nn.Module (PyTorch)

IMPORTANT: training method is named fit() not train() to avoid
conflicting with PyTorch's built-in nn.Module.train() method.

Architecture: 2-layer GCN following Kipf & Welling (ICLR 2017)
  Input(d) -> GCNLayer(64) -> ReLU -> Dropout(0.5) -> GCNLayer(C)

Each GCN layer computes: H' = A_hat * H * W
where A_hat = D^{-1/2} * (A + I) * D^{-1/2}  (pre-computed in loader)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from code.base_class.method import method


# -------------------------------------------------------
# Single GCN layer
# -------------------------------------------------------
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        # x * W then A_hat * result
        out = torch.spmm(adj, torch.mm(x, self.weight))
        return out + self.bias


# -------------------------------------------------------
# 2-layer GCN network
# -------------------------------------------------------
class GCNNetwork(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, dropout):
        super(GCNNetwork, self).__init__()
        self.gc1     = GraphConvLayer(n_features, n_hidden)
        self.gc2     = GraphConvLayer(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# -------------------------------------------------------
# Method class (course template style)
# -------------------------------------------------------
class Method_GCN_Node_Classification(method, nn.Module):

    # hyper-parameters
    learning_rate = 0.01
    weight_decay  = 5e-4
    n_hidden      = 64
    dropout       = 0.5
    max_epoch     = 200

    # set by the script before calling run()
    data          = None
    dataset_name  = 'unknown'
    result_folder = 'result/stage_5_result'

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    # --------------------------------------------------
    # forward pass (required by nn.Module)
    # --------------------------------------------------
    def forward(self, x, adj):
        return self.network(x, adj)

    # --------------------------------------------------
    # accuracy helper
    # --------------------------------------------------
    def accuracy(self, logits, labels, idx):
        preds = logits[idx].max(1)[1]
        return preds.eq(labels[idx]).float().mean().item()

    # --------------------------------------------------
    # fit() — renamed from train() to avoid PyTorch conflict
    # --------------------------------------------------
    def fit(self):
        adj    = self.data['utility']['A']
        X      = self.data['X']
        y      = self.data['y']
        idx_tr = self.data['idx_train']
        idx_va = self.data['idx_val']

        n_features = X.shape[1]
        n_classes  = int(y.max().item()) + 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        adj, X, y = adj.to(device), X.to(device), y.to(device)
        idx_tr    = idx_tr.to(device)
        idx_va    = idx_va.to(device)

        # build network
        self.network = GCNNetwork(n_features, self.n_hidden, n_classes, self.dropout).to(device)

        optimizer = optim.Adam(self.network.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_loss_list, val_loss_list = [], []
        train_acc_list,  val_acc_list  = [], []

        print('--start training...')
        t0 = time.time()

        for epoch in range(self.max_epoch):
            self.network.train()
            optimizer.zero_grad()
            logits   = self.network(X, adj)
            loss_tr  = criterion(logits[idx_tr], y[idx_tr])
            loss_tr.backward()
            optimizer.step()

            self.network.eval()
            with torch.no_grad():
                logits  = self.network(X, adj)
                loss_va = criterion(logits[idx_va], y[idx_va]).item()

            acc_tr = self.accuracy(logits, y, idx_tr)
            acc_va = self.accuracy(logits, y, idx_va)

            train_loss_list.append(loss_tr.item())
            val_loss_list.append(loss_va)
            train_acc_list.append(acc_tr)
            val_acc_list.append(acc_va)

            if epoch % 20 == 0:
                print('Epoch: {:3d} | Train Loss: {:.4f} Acc: {:.4f} | '
                      'Val Loss: {:.4f} Acc: {:.4f}'.format(
                          epoch, loss_tr.item(), acc_tr, loss_va, acc_va))

        print('--training done, time: {:.1f}s'.format(time.time() - t0))

        self._X   = X
        self._y   = y
        self._adj = adj

        return {
            'train_loss': train_loss_list,
            'val_loss':   val_loss_list,
            'train_acc':  train_acc_list,
            'val_acc':    val_acc_list,
        }

    # --------------------------------------------------
    # test
    # --------------------------------------------------
    def test(self):
        idx_te = self.data['idx_test'].to(self.device)

        self.network.eval()
        with torch.no_grad():
            logits = self.network(self._X, self._adj)

        pred_y = logits[idx_te].max(1)[1].cpu()
        true_y = self._y[idx_te].cpu()
        return pred_y, true_y

    # --------------------------------------------------
    # plot learning curves
    # --------------------------------------------------
    def plot_curves(self, history):
        os.makedirs(self.result_folder, exist_ok=True)
        epochs = range(self.max_epoch)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Learning Curves - {}'.format(self.dataset_name), fontsize=13)

        ax1.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
        ax1.plot(epochs, history['val_loss'],   label='Val Loss',   color='coral', linestyle='--')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Cross-Entropy Loss'); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, history['train_acc'], label='Train Acc', color='steelblue')
        ax2.plot(epochs, history['val_acc'],   label='Val Acc',   color='coral', linestyle='--')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.result_folder,
                            'learning_curve_{}.png'.format(self.dataset_name))
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print('Curve saved to:', path)

    # --------------------------------------------------
    # run() -- called by the script
    # --------------------------------------------------
    def run(self):
        print('method:{}, dataset:{}'.format(self.method_name, self.dataset_name))

        history          = self.fit()
        pred_y, true_y   = self.test()
        self.plot_curves(history)

        return pred_y, true_y
