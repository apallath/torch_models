# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from pinn_model import PINN

"""
DE: u''(x) - u(x) = f(x)
f(x) = -(pi^2 + 1) sin(pi x)
"""

Nfs = [50, 100, 200, 500]

L2es = []
for Nf in Nfs:
    fname = "PINN_{}.pt".format(Nf)
    pn = torch.load(fname)
    x_test = np.linspace(-1,1,100).reshape(-1,1)
    x_test = torch.tensor(x_test).type(torch.FloatTensor)
    # L2 approximation error
    L2e = pn.MSE(pn(x_test), torch.sin(np.pi * x_test))
    L2es.append(L2e)

plt.figure(dpi=150)
plt.plot(np.array(Nfs), np.array(L2es), 'o-')
plt.savefig("L2error_comp.png")
plt.show()
