"""@author Akash Pallath
PyTorch autoencoder plots code
Dataset: MNIST
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from autoencoder import AE, AE_model

import pickle #for loading saved model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="File containing trained Autoencoder model (default: trained_convnet.pkl)")
parser.add_argument("--show", action="store_true", help="Show interactive matplotlib plots")
parser.add_argument("--lossoutfile", help="File to save output image of loss history to (default: loss_history_<num_epochs>.png")
parser.add_argument("--recoutfile", help="File to save output image of reconstructions to (default: rec_<num_epochs>.png")

args = parser.parse_args()
if args.file is None:
    filename = "trained_convnet.pkl"
else:
    filename = args.file

f = open(filename, 'rb')
ae = pickle.load(f)
f.close()

if args.lossoutfile is None:
    imgfile = "loss_history_{}.png".format(ae.last_epoch)
else:
    imgfile = args.lossoutfile


loss_history = np.array(ae.loss_history)
test_loss_history = np.array(ae.test_loss_history)

plt.figure(dpi=150)
ax = plt.gca()
ax.plot(loss_history[:,0], loss_history[:,1], 'x-', label="Training loss")
ax.plot(test_loss_history[:,0], test_loss_history[:,1], '+-', label="Test loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("MSE loss")
ax.legend()
ax.set_title("Losses after {} epochs [{:.1f}s]".format(ae.last_epoch, ae.train_time))
plt.savefig(imgfile)
if args.show:
    plt.show()


"""Plot reconstructions"""
if args.recoutfile is None:
    recfile = "rec_{}.png".format(ae.last_epoch)
else:
    recfile = args.recoutfile

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
rec_dataset = torchvision.datasets.MNIST(root="./dataset", train=False, transform=transform, download=True)
rec_loader = torch.utils.data.DataLoader(rec_dataset, batch_size=10, shuffle=False)
rec_examples = None

with torch.no_grad():
    for (features, labels) in rec_loader:
        rec_examples = features.view(-1, 784)
        reconstruction = ae.net(rec_examples)

        # break after loading a single batch
        break

    nexamples = 10
    plt.figure(figsize=(20, 4))
    for idx in range(nexamples):
        # Original images
        ax = plt.subplot(2, nexamples, idx + 1)
        plt.imshow(rec_examples[idx].numpy().reshape(28, 28))
        plt.gray()

        # Reconstructed images
        ax = plt.subplot(2, nexamples, idx + 1 + nexamples)
        plt.imshow(reconstruction[idx].numpy().reshape(28, 28))
        plt.gray()

    plt.savefig(recfile)
    if args.show:
        plt.show()
