import numpy as np
import matplotlib.pyplot as plt

from cnn_model import CNN, ConvNet

import pickle #for loading saved model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="File containing trained CNN model")
parser.add_argument("--show", action="store_true")
parser.add_argument("--outfile", help="File to save output image to (default: loss_history.png")

args = parser.parse_args()
if args.file is None:
    filename = "trained_convnet.pkl"
else:
    filename = args.file

if args.outfile is None:
    imgfile = "loss_history.png"
else:
    imgfile = args.outfile

f = open(filename, 'rb')
cn = pickle.load(f)
f.close()

loss_history = np.array(cn.loss_history)
test_loss_history = np.array(cn.test_loss_history)

plt.figure(dpi=150)
ax = plt.gca()
ax.plot(loss_history[:,0], loss_history[:,1], 'x-', label="Training loss")
ax.plot(test_loss_history[:,0], test_loss_history[:,1], '+-', label="Test loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Binary cross-entropy loss")
ax.legend()
tr_acc = cn.train_acc(128)
ts_acc = cn.test_acc(128)
ax.set_title("After {} epochs [{:.1f}s]: \n Train accuracy = {:.2f} %, test accuracy = {:.2f} %".format(cn.last_epoch, cn.train_time, tr_acc, ts_acc))
plt.savefig(imgfile)
if args.show:
    plt.show()
