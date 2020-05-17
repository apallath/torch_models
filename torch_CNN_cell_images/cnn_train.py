import numpy as np
import torch

from cnn_model import CNN, ConvNet

import pickle #for saving models
import argparse #for passing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("epochs", help="Number of epochs to train neural network for")
parser.add_argument("batch_size", help="Batch size")
parser.add_argument("--outfile", help="File to save trained model to (default: trained_convnet.pkl)")
parser.add_argument("--restart", action="store_true", help="Restart from existing saved model")
parser.add_argument("--restartfile", help="File containing trained model to restart from (default: trained_convnet.pkl)")
parser.add_argument("--debug", action="store_true", help="Use a smaller training set (1/10) for debugging")

args = parser.parse_args()
if args.outfile is None:
    ofilename = "trained_convnet.pkl"
else:
    ofilename = args.outfile

if args.restartfile is None:
    rfilename = "trained_convnet.pkl"
else:
    rfilename = args.restartfile

if args.restart == False:
    cn = ConvNet(debug=args.debug)
    cn.train(num_epochs = int(args.epochs), batch_size = int(args.batch_size))
    f = open(ofilename, 'wb')
    pickle.dump(cn, f)
    f.close()
else:
    f = open(rfilename, 'rb')
    cn = pickle.load(f)
    f.close()
    cn.train(num_epochs = int(args.epochs), batch_size = int(args.batch_size))
    f = open(ofilename, 'wb')
    pickle.dump(cn, f)
    f.close()
