import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import timeit

from torch.utils.data import Dataset, DataLoader
from dataloader import Cell_dataset

# Define CNN architecture and forward pass
# Architecture inspired by AlexNet
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #conv-pool-layer 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=10, stride=2, padding=4)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        #conv-pool-layer 2
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)
        #conv-pool-layer 3
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(2)
        #fc-layer 1
        self.fc1 = torch.nn.Linear(64*12*12, 64)
        self.fcrelu1 = torch.nn.ReLU()
        #fc-layer 2
        self.fc2 = torch.nn.Linear(64, 16)
        self.fcrelu2 = torch.nn.ReLU()
        #fc-output-layer
        self.fc = torch.nn.Linear(16, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        #conv layers
        out = self.conv1(x); #print(out.size())
        out = self.bn1(out); #print(out.size())
        out = self.relu1(out); #print(out.size())
        out = self.pool1(out); #print(out.size())

        out = self.conv2(out); #print(out.size())
        out = self.bn2(out); #print(out.size())
        out = self.relu2(out); #print(out.size())
        out = self.pool2(out); #print(out.size())

        out = self.conv3(out); #print(out.size())
        out = self.bn3(out); #print(out.size())
        out = self.relu3(out); #print(out.size())
        out = self.pool3(out); #print(out.size())

        #linear layers
        out = out.view(out.size(0), -1); #print(out.size())

        out = self.fc1(out); #print(out.size())
        out = self.fcrelu1(out); #print(out.size())

        out = self.fc2(out); #print(out.size())
        out = self.fcrelu2(out); #print(out.size())

        #final output layer
        out = self.fc(out); #print(out.size())
        out = self.act(out); #print(out.size())

        return out

class ConvNet:
    # Initialize the class
    def __init__(self, debug=False):
        # Hyperparameters
        self.LRATE = 1e-3

        # Load train dataset
        if(debug):
            self.train_data = Cell_dataset('train_small/')
            self.test_data = Cell_dataset('test_small/')
        else:
            self.train_data = Cell_dataset('train/')
            self.test_data = Cell_dataset('test/')

        # Define architecture and initialize
        self.net = CNN()

        # Define the loss function
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LRATE)

        # Epoch stopped at
        self.last_epoch = 0

        # Loss history
        self.loss_history = []
        self.test_loss_history = []

        # Training time
        self.train_time = 0

    # Trains the model by minimizing the Cross Entropy loss
    def train(self, num_epochs = 10, batch_size = 128):
        # Create a PyTorch data loader object
        train_loader = torch.utils.data.DataLoader(self.train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        start_time = timeit.default_timer()

        loss_history = []
        test_loss_history = []

        for epoch in range(num_epochs):
            epochlosses = []
            testepochlosses = []

            for it, (images, labels) in enumerate(train_loader):
                # Reset gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.net(images)
                # Compute loss
                loss = self.loss_fn(outputs, labels)
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()

                elapsed = timeit.default_timer() - start_time
                start_time = timeit.default_timer()
                self.train_time += elapsed

                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs'
                       %(epoch + self.last_epoch + 1, self.last_epoch + num_epochs, \
                       it+1, int(np.ceil(len(self.train_data)/batch_size)), loss.cpu().data, elapsed))

                epochlosses.append(loss.data.numpy())

            # Calculate epoch test loss
            for it, (images, labels) in enumerate(test_loader):
                outputs = self.net(images)
                # Compute loss
                loss = self.loss_fn(outputs, labels)
                testepochlosses.append(loss.data.numpy())

            epochloss = np.mean(np.array(epochlosses))
            testepochloss = np.mean(np.array(testepochlosses))

            loss_history.append([epoch + self.last_epoch + 1, epochloss])
            test_loss_history.append([epoch + self.last_epoch + 1, testepochloss])
            print("Epoch [{}/{}], Training loss: {:.4f}, Testing loss = {:.4f}".format(\
                epoch + self.last_epoch + 1, self.last_epoch + num_epochs, epochloss, testepochloss))

        self.last_epoch += num_epochs

        self.train_acc(batch_size)
        self.test_acc(batch_size)
        # Store loss history as list for plots
        self.loss_history.extend(loss_history)
        self.test_loss_history.extend(test_loss_history)

    def train_acc(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        # Train prediction accuracy
        correct = 0
        total = 0
        for images, labels in train_loader:
            outputs = self.net(images)
            predicted = torch.where(outputs < 0.5, torch.tensor(0), torch.tensor(1))
            total += labels.size(0)
            correct += int((predicted == labels).sum())
        print("Train Accuracy: {:.5f} %".format(100.0 * correct / total))
        return 100.0 * correct / total

    def test_acc(self, batch_size):
        # Create a PyTorch data loader object
        test_loader = torch.utils.data.DataLoader(self.test_data,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        # Test prediction accuracy
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = self.net(images)
            predicted = torch.where(outputs < 0.5, torch.tensor(0), torch.tensor(1))
            total += labels.size(0)
            correct += int((predicted == labels).sum())
        print("Test Accuracy: {:.5f} %".format(100.0 * correct / total))
        return 100.0 * correct / total
