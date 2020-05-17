import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN,self).__init__()
        #layer definitions
        self.FC1 = nn.Linear(1,50)
        self.act1 = nn.Tanh()
        self.FC2 = nn.Linear(50,50)
        self.act2 = nn.Tanh()
        self.FC3 = nn.Linear(50,1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.FC1(x)
        x2 = self.act1(x1)
        x3 = self.FC2(x2)
        x4 = self.act2(x3)
        x5 = self.FC3(x4)
        u = self.tanh(x5)
        return u

    # loss function
    def MSE(self,ypred,ytrue):
        return torch.mean((ypred - ytrue)**2)

    # Glorot initialization of weight matrix
    def glorot_init_mat(self,shape):
        din = shape[0]
        dout = shape[1]
        var = torch.tensor([2.0/(din+dout)])
        std = torch.sqrt(var)
        mean = torch.tensor([0.0])
        dist = torch.distributions.normal.Normal(mean, std)
        return dist.sample(shape)
