'''
Concrete MethodModule class for a specific learning MethodModule
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
import numpy as np





class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model, setting to 30 since MNIST dataset is extremely large
    max_epoch = 30
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    training_history = None
    plot_destination_folder_path = None
    plot_file_name = 'MLP_convergence_curve.png'
    last_plot_path = None

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # 28 x 28 pixel images which equal 784 input features
        #Choosing about 1/3 of input size for first hidden layer neurons(256)
        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU() #activation function
        #Send the neuron outputs into the second hidden layer
        self.fc_layer_2 = nn.Linear(256, 128)
        # Using ReLU again for the second activation function
        self.activation_func_2 = nn.ReLU()
        #Finally we send the output to the output later
        #Not using softmax activation func bc crossentropyloss function will be use in train
        #linear fully connected output layer
        self.fc_layer_3 = nn.Linear(128, 10) #Digits 0-9
        #Current Network shape: input(784) -> hiddenlayer1(256)->ReLU->hiddenlayer2(128)->ReLU->FC(10)
        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}



    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        #pass through layer 1 and apply ReLU
        h1 = self.activation_func_1(self.fc_layer_1(x))
        #pass that value through layer 2 and apply ReLU
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        #pass through the output layer for y prediction
        #Shape: (batch_size, 10)
        y_pred = self.fc_layer_3(h2)

        return y_pred


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def save_convergence_plot(self):
        if not self.training_history['epoch']:
            return None

        if self.plot_destination_folder_path is None:
            project_root = Path(__file__).resolve().parents[2]
            destination_dir = project_root / 'result' / 'stage_2_result'
        else:
            destination_dir = Path(self.plot_destination_folder_path)

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / self.plot_file_name

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(self.training_history['epoch'], self.training_history['loss'], color='tab:red', linewidth=2)
        axes[0].set_title('MLP Convergence Curve')
        axes[0].set_ylabel('Training Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.training_history['epoch'], self.training_history['accuracy'], color='tab:blue', linewidth=2)
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
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            train_metrics = accuracy_evaluator.evaluate()
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(train_loss.item())
            self.training_history['accuracy'].append(train_metrics['accuracy'])

            #Track every 5 epochs (6 checkpoints)
            if epoch%5 == 0:
                print('Epoch:', epoch, 'Accuracy:', train_metrics['accuracy'], 'Loss:', train_loss.item())

        return self.training_history
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
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
            
