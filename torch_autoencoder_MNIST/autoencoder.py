"""@author Akash Pallath
PyTorch autoencoder class (AE_module) with ability to:
- restart training from saved state
- store train and test loss history
- save trained module as TorchScript
Dataset: MNIST
"""

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import timeit

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Encoder layers
        self.enc1L = nn.Linear(784, 392)
        self.enc1A = nn.ReLU()
        self.enc2L = nn.Linear(392, 196)
        self.enc2A = nn.ReLU()
        self.enc3L = nn.Linear(196, 98)
        self.enc3A = nn.ReLU()
        self.enc4L = nn.Linear(98, 16)
        self.enc4A = nn.ReLU()

        # Decoder layers
        self.dec1L = nn.Linear(16, 98)
        self.dec1A = nn.ReLU()
        self.dec2L = nn.Linear(98, 196)
        self.dec2A = nn.ReLU()
        self.dec3L = nn.Linear(196, 392)
        self.dec3A = nn.ReLU()
        self.dec4L = nn.Linear(392, 784)
        self.dec4A = nn.ReLU()

    def encoder(self,x):
        z = self.enc1L(x)
        z = self.enc1A(z)
        z = self.enc2L(z)
        z = self.enc2A(z)
        z = self.enc3L(z)
        z = self.enc3A(z)
        z = self.enc4L(z)
        z = self.enc4A(z)
        return z

    def decoder(self,z):
        x = self.dec1L(z)
        x = self.dec1A(x)
        x = self.dec2L(x)
        x = self.dec2A(x)
        x = self.dec3L(x)
        x = self.dec3A(x)
        x = self.dec4L(x)
        x = self.dec4A(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AE_model:
    def __init__(self):
        # Hyperparameters
        self.LRATE = 1e-3

        # Train and test datasets
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.train_data = torchvision.datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
        self.test_data = torchvision.datasets.MNIST(root="./dataset", train=False, transform=transform, download=True)

        # Autoencoder architecture
        self.net = AE()

        # Loss function
        self.loss_fn = nn.MSELoss(reduction="mean")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LRATE)

        # Epoch stopped at
        self.last_epoch = 0

        # Loss history
        self.loss_history = []
        self.test_loss_history = []

        # Training time
        self.train_time = 0

    def train(self, num_epochs = 10, batch_size = 128):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

        # Record time elapsed
        start_time = timeit.default_timer()

        loss_history = []
        test_loss_history = []

        for epoch in range(num_epochs):
            epochlosses = []
            testepochlosses = []

            for it, (features, labels) in enumerate(train_loader):
                # Reset gradients
                self.optimizer.zero_grad()
                # Reshape input for neural net
                features = features.view(-1, 784)
                # Forward pass
                outputs = self.net(features)
                # Compute loss
                loss = self.loss_fn(outputs, features)
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()

                elapsed = timeit.default_timer() - start_time
                start_time = timeit.default_timer()
                self.train_time += elapsed

                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs'
                       %(epoch + self.last_epoch + 1, self.last_epoch + num_epochs, \
                       it+1, int(np.ceil(len(self.train_data)/batch_size)), loss.data, elapsed))

                epochlosses.append(loss.data.numpy())

            # Average loss over the epoch
            epochloss = np.mean(np.array(epochlosses))
            loss_history.append([epoch + self.last_epoch + 1, epochloss])

            print("Epoch [{}/{}], Training loss: {:.4f}".format(\
                epoch + self.last_epoch + 1, self.last_epoch + num_epochs, epochloss))

            # Calculate epoch test loss every X epochs
            if epoch % 1 == 0:
                for it, (features, labels) in enumerate(test_loader):
                    # Reshape input for neural net
                    features = features.view(-1, 784)
                    # Forward pass
                    outputs = self.net(features)
                    # Compute loss
                    loss = self.loss_fn(outputs, features)
                    # Update test losses
                    testepochlosses.append(loss.data.numpy())

                testepochloss = np.mean(np.array(testepochlosses))
                test_loss_history.append([epoch + self.last_epoch + 1, testepochloss])

                print("Epoch [{}/{}], Testing loss = {:.4f}".format(\
                    epoch + self.last_epoch + 1, self.last_epoch + num_epochs, testepochloss))

        self.last_epoch += num_epochs

        # Store loss history as list for plots
        self.loss_history.extend(loss_history)
        self.test_loss_history.extend(test_loss_history)
