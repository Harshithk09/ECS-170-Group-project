from code.base_class.method import method
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 15
    learning_rate = 1e-3

#Ablation 3:
    def __init__(self, mName, mDescription, in_channels, num_classes):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Layer 1 (Back to baseline 3x3)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2 (Back to baseline 3x3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # --- ADDED FOR ABLATION 3 ---
        self.dropout = nn.Dropout(0.5)
        # ----------------------------

        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # --- ADDED FOR ABLATION 3 ---
        x = self.dropout(x)
        # ----------------------------

        x = self.fc(x)
        return x

###########################################################################

#Ablation 2:
    # def __init__(self, mName, mDescription, in_channels, num_classes):
    #     method.__init__(self, mName, mDescription)
    #     nn.Module.__init__(self)
    #
    #     # --- CHANGED FOR ABLATION 2 ---
    #     # Layer 1 (kernel_size=5, padding=2)
    #     self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
    #     self.relu1 = nn.ReLU()
    #     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    #     # Layer 2 (kernel_size=5, padding=2)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
    #     self.relu2 = nn.ReLU()
    #     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # ------------------------------
    #
    #     self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
    #
    #     # Reverted back to 32 (baseline) instead of 64
    #     self.fc = nn.Linear(32 * 4 * 4, num_classes)
    #
    # def forward(self, x):
    #     x = self.pool1(self.relu1(self.conv1(x)))
    #     x = self.pool2(self.relu2(self.conv2(x)))
    #
    #     x = self.adaptive_pool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x

###########################################################################

#Ablation 1:
    # def __init__(self, mName, mDescription, in_channels, num_classes):
    #     method.__init__(self, mName, mDescription)
    #     nn.Module.__init__(self)
    #
    #     # Layer 1
    #     self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
    #     self.relu1 = nn.ReLU()
    #     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    #     # Layer 2
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    #     self.relu2 = nn.ReLU()
    #     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #
    #     # --- ADDED FOR ABLATION 1 ---
    #     # Layer 3
    #     self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.relu3 = nn.ReLU()
    #     self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # ----------------------------
    #
    #     self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
    #
    #     # UPDATE THIS LINE FOR ABLATION 1: Change 32 to 64
    #     self.fc = nn.Linear(64 * 4 * 4, num_classes)
    #
    # def forward(self, x):
    #     x = self.pool1(self.relu1(self.conv1(x)))
    #     x = self.pool2(self.relu2(self.conv2(x)))
    #
    #     # --- ADDED FOR ABLATION 1 ---
    #     x = self.pool3(self.relu3(self.conv3(x)))
    #     # ----------------------------
    #
    #     x = self.adaptive_pool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x

###########################################################################

# Baseline:
#     def __init__(self, mName, mDescription, in_channels, num_classes):
#         method.__init__(self, mName, mDescription)
#         nn.Module.__init__(self)
#
#         # BASELINE: 2 Convolutional Layers with 3x3 kernels
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # This standardizes the image size before the linear layer
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
#
#         # Fully connected layer
#         self.fc = nn.Linear(32 * 4 * 4, num_classes)
#
#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        # Normalizes pixels and uses mini-batches to save RAM
        tensor_X = torch.FloatTensor(np.array(X)) / 255.0
        tensor_y = torch.LongTensor(np.array(y))

        dataset = TensorDataset(tensor_X, tensor_y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(self.max_epoch):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                y_pred = self.forward(batch_X)
                loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch: {epoch} | Loss: {total_loss / len(dataloader):.4f}')

    def test(self, X):
        tensor_X = torch.FloatTensor(np.array(X)) / 255.0
        dataset = TensorDataset(tensor_X)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch_X in dataloader:
                y_pred = self.forward(batch_X[0])
                all_preds.extend(y_pred.max(1)[1].tolist())
        return np.array(all_preds)

    def run(self):
        print('--start training...')
        self.train(self.data['X_train'], self.data['y_train'])
        print('--start testing...')
        pred_y = self.test(self.data['X_test'])
        return {'pred_y': pred_y, 'true_y': self.data['y_test']}
