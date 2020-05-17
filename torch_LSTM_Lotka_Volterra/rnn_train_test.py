"""Train LSTM and vanilla RNN for one-step-ahead prediction on Lotka-Volterra equation data
   Plot predictions and calculate errors
   Compare LSTM prediction to vanilla RNN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

from rnn_models import *

alpha = 1.0
beta = 0.1
gamma = 1.5
sigma = 0.75

t = np.linspace(0,60,2000)

"""Lotka-Volterra Equations"""

#store [x,y] in a single vector X
#x = X[0], y = X[1]

#dX/dt = F[X,t]
def F(X, t):
    return np.array([X[0] * (alpha - beta * X[1]), -X[1] * (gamma - sigma * X[0])])

#Initial condition X0
X0 = np.array([10.0, 5.0])

#solve for X = [x,y] using SciPy
X = scipy.integrate.odeint(F, X0, t)

#Plot
plt.figure(dpi=150)
ax=plt.gca()
ax.plot(t,X[:,0],label="x (population of prey)")
ax.plot(t,X[:,1],label="y (population of predators)")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Predator/prey population")
plt.savefig("original_data.png")
#plt.show()

"""Train-test split and lagged data generation"""

#train-test-split
t_train = t[:int(2/3*2000)]
X_train = X[:int(2/3*2000),:]
ntrain = len(t_train)

t_test = t[int(2/3*2000):]
X_test = X[int(2/3*2000):,:]
ntest = len(t_test)

#data normalization
mu = np.mean(X_train,axis=0)
sigma = np.std(X_train,axis=0)
X_train = (X_train - mu)/sigma
X_test = (X_test - mu)/sigma

#Plot normalized data
plt.figure(dpi=150)
ax=plt.gca()
ax.plot(t_train,X_train[:,0],label="x (train)")
ax.plot(t_train,X_train[:,1],label="y (train)")
ax.axvline(t[int(2/3*2000)],color="black")
ax.plot(t_test,X_test[:,0],label="x (test)")
ax.plot(t_test,X_test[:,1],label="y (test)")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Predator/prey population")
plt.savefig("original_data_normalized.png")
#plt.show()

#Generate lagged training data
nlags = 8
nfeatures = 2
X_train_lag = np.zeros((nlags,ntrain-nlags,nfeatures))
Y_train = X_train[nlags:,:]
for s in range(ntrain-nlags):
    for lag in range(nlags-1,-1,-1):
        X_train_lag[lag,s,:] = X_train[s+nlags-lag-1,:]

#Generate lagged test data
nlags = 8
nfeatures = 2
X_test_lag = np.zeros((nlags,ntest-nlags,nfeatures))
Y_test = X_test[nlags:,:]
for s in range(ntest-nlags):
    for lag in range(nlags-1,-1,-1):
        X_test_lag[lag,s,:] = X_test[s+nlags-lag-1,:]

"""Train RNN"""
MAX_EPOCHS = 5000
BATCH_SIZE = 128
hidden_dim = 20

print("Training RNN")
rnn = RNN(X_train_lag, Y_train, hidden_dim)
rnn.train(MAX_EPOCHS, BATCH_SIZE)

#plot loss history
plt.figure(dpi=150)
ax=plt.gca()
ax.plot(np.array(rnn.training_loss))
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("RNN training loss")
plt.savefig("rnn_loss_history_{}_lags.png".format(nlags))
#plt.show()

#train prediction
Y_train_pred = rnn.predict(X_train_lag)

plt.figure(dpi=150)
ax=plt.gca()
ax.plot(t_train[nlags:],Y_train[:,0],':',label="x (population of prey) [true]")
ax.plot(t_train[nlags:],Y_train[:,1],':',label="y (population of predators) [true]")
ax.plot(t_train[nlags:],Y_train_pred[:,0],label="x (population of prey) [predicted]")
ax.plot(t_train[nlags:],Y_train_pred[:,1],label="y (population of predators) [predicted]")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Predator/prey population")
ax.set_title("Training set RNN prediction")
plt.savefig("rnn_train_prediction_{}_lags_normalized.png".format(nlags))
#plt.show()

#test one-step-ahead prediction
data = X_test_lag[:,-1,:].reshape(nlags,1,nfeatures)
Y_test_pred = rnn.one_step_ahead(data, len(t_test)-nlags)
#de-normalize
X_test_denorm = sigma * X_test[nlags:] + mu
Y_test_pred_denorm = sigma * Y_test_pred + mu
plt.figure(dpi=150)
ax=plt.gca()
ax.plot(t_test[nlags:],X_test_denorm[:,0],':',label="x (population of prey) [true]")
ax.plot(t_test[nlags:],X_test_denorm[:,1],':',label="y (population of predators) [true]")
ax.plot(t_test[nlags:],Y_test_pred_denorm[:,0],label="x (population of prey) [predicted]")
ax.plot(t_test[nlags:],Y_test_pred_denorm[:,1],label="y (population of predators) [predicted]")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Predator/prey population")
ax.set_title("Test set RNN prediction")
plt.savefig("rnn_test_prediction_{}_lags.png".format(nlags))
#Norms
error = rnn.l2_norm_error(X_test_denorm, Y_test_pred_denorm)
print("(RNN) L2 norm error for x: {:.4f}, L2 norm error for y: {:.4f}".format(error[0], error[1]))
#plt.show()

"""Train LSTM"""
MAX_EPOCHS = 2000
BATCH_SIZE = 128
hidden_dim = 20

print("Training LSTM")
lstm = LSTM(X_train_lag, Y_train, hidden_dim)
lstm.train(MAX_EPOCHS, BATCH_SIZE)

#plot loss history
plt.figure(dpi=150)
ax=plt.gca()
ax.plot(np.array(lstm.training_loss))
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("LSTM training loss")
plt.savefig("lstm_loss_history_{}_lags.png".format(nlags))
#plt.show()

Y_train_pred = lstm.predict(X_train_lag)

plt.figure(dpi=150)
ax=plt.gca()
ax.plot(t_train[nlags:],Y_train[:,0],':',label="x (population of prey) [true]")
ax.plot(t_train[nlags:],Y_train[:,1],':',label="y (population of predators) [true]")
ax.plot(t_train[nlags:],Y_train_pred[:,0],label="x (population of prey) [predicted]")
ax.plot(t_train[nlags:],Y_train_pred[:,1],label="y (population of predators) [predicted]")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Predator/prey population")
ax.set_title("Training set LSTM prediction")
plt.savefig("lstm_train_prediction_{}_lags_normalized.png".format(nlags))
#plt.show()

#test one-step-ahead prediction
data = X_test_lag[:,-1,:].reshape(nlags,1,nfeatures)
Y_test_pred = lstm.one_step_ahead(data, len(t_test)-nlags)
#de-normalize
X_test_denorm = sigma * X_test[nlags:] + mu
Y_test_pred_denorm = sigma * Y_test_pred + mu
plt.figure(dpi=150)
ax=plt.gca()
ax.plot(t_test[nlags:],X_test_denorm[:,0],':',label="x (population of prey) [true]")
ax.plot(t_test[nlags:],X_test_denorm[:,1],':',label="y (population of predators) [true]")
ax.plot(t_test[nlags:],Y_test_pred_denorm[:,0],label="x (population of prey) [predicted]")
ax.plot(t_test[nlags:],Y_test_pred_denorm[:,1],label="y (population of predators) [predicted]")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Predator/prey population")
ax.set_title("Test set LSTM prediction")
plt.savefig("lstm_test_prediction_{}_lags.png".format(nlags))
#Norms
error = np.sqrt(np.mean((X_test_denorm - Y_test_pred_denorm)**2, axis = 0))
print("(LSTM) L2 norm error for x: {:.4f}, L2 norm error for y: {:.4f}".format(error[0], error[1]))
#plt.show()
