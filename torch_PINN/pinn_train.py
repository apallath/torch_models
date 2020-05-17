# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyDOE
from pinn_model import PINN

import argparse

import argparse #for passing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("Nf", help="Number of f-points to solve problem for")
parser.add_argument("epochs", help="Number of epochs to train neural network for")
args = parser.parse_args()

"""data prep"""

"""
DE: u''(x) - u(x) = f(x)
f(x) = -(pi^2 + 1) sin(pi x)
"""

# %%
# Generate samples for x, f
Nf = int(args.Nf)
design = pyDOE.lhs(1, samples = Nf)
x = -1 + 2 * design[:,0]
f = -(np.pi**2 + 1)*np.sin(np.pi*x)

# plot function
plt.figure(dpi=150)
plt.scatter(x,f)
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('f')
#plt.show()

# %%
# Generate samples for x, u [boundary conditions]
Nu = 2
x_bc = np.array([-1.0, 1.0])
u_bc = np.array([0.0,0.0])


"""NN class, init, loss functions, dataloader"""

# %%
"""training process"""

# %%
# load f data into torch
x = x.reshape((-1, 1))
x = torch.tensor(x).type(torch.FloatTensor)
f = f.reshape((-1, 1))
f = torch.tensor(f).type(torch.FloatTensor)

#normalize f
mu = f.mean()
sigma = f.std()
f = (f - mu)/sigma

# load BC data into torch
x_bc = x_bc.reshape((-1, 1))
x_bc = torch.tensor(x_bc).type(torch.FloatTensor)
u_bc = u_bc.reshape((-1, 1))
u_bc = torch.tensor(u_bc).type(torch.FloatTensor)

# %%
#Instantiate class
pn = PINN()

#Initialize weights
#Glorot initialization
[W, b] = list(pn.FC1.parameters())
W = pn.glorot_init_mat(W.shape)
b.data.fill_(0)

[W, b] = list(pn.FC2.parameters())
W = pn.glorot_init_mat(W.shape)
b.data.fill_(0)

[W, b] = list(pn.FC3.parameters())
W = pn.glorot_init_mat(W.shape)
b.data.fill_(0)

# %%
# Perform backprop
MAX_EPOCHS = int(args.epochs)
LRATE = 3e-4

#Use Adam for training
optimizer = torch.optim.Adam(pn.parameters(), lr=LRATE)

loss_history_u = []
loss_history_f = []
loss_history = []

for epoch in range(MAX_EPOCHS):
    #full batch

    #u
    upred_bc = pn(x_bc)
    mse_u = pn.MSE(upred_bc, u_bc)
    loss_history_u.append([epoch, mse_u])

    #f
    xc = x.clone()
    xc.requires_grad = True
    upred = pn(xc)
    upred1 = torch.autograd.grad(upred.sum(),xc,create_graph=True)[0]
    upred2 = torch.autograd.grad(upred1.sum(),xc,create_graph=True)[0]
    mse_f = pn.MSE(upred2 - upred, sigma * f + mu) #rescale f before computing loss
    loss_history_f.append([epoch, mse_f])

    loss = mse_u + mse_f
    loss_history.append([epoch, loss])

    #optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print("Epoch: {}, MSE_u: {:.4f}, MSE_f: {:.4f}, MSE: {:.4f}".format((epoch+1), mse_u, mse_f, loss))

#%%
#save trained model
fname = "PINN_{}.pt".format(Nf)
torch.save(pn, fname)

#%%
loss_history = np.array(loss_history)
plt.figure(dpi=150)
plt.plot(loss_history[:,0], loss_history[:,1])
plt.savefig("PINN_loss_hist_{}.png".format(Nf))
#plt.show()

#%%
plt.figure(dpi=150)
x_test = np.linspace(-1,1,100).reshape(-1,1)
x_test = torch.tensor(x_test).type(torch.FloatTensor)
plt.scatter(x_test.data.numpy(), pn(x_test).data.numpy(), label="Neural network solution")
plt.scatter(x_test.data.numpy(), np.sin(np.pi * x_test.data.numpy()), label="True solution")
plt.legend()
plt.savefig("PINN_soln_comp_{}.png".format(Nf))
#plt.show()
