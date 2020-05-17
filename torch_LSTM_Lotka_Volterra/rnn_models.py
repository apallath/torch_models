import numpy as np
import torch
import timeit

# Vanilla RNN
class RNN:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):

        self.dtype = torch.FloatTensor
        # X has the form nlags x samples x nfeatures
        # Y has the form samples x nfeatures

        # Define PyTorch variables
        self.X = torch.from_numpy(X).type(self.dtype)
        self.Y = torch.from_numpy(Y).type(self.dtype)
        self.X.requires_grad = False
        self.Y.requires_grad = False

        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]
        self.samples = X.shape[1]

        # RNN update rule
        # H = tanh(XU + b + XW) (recursively update for each lag)
        # Y = HV + c
        # Parameters: U, b, W, V, c

        # Initialize network weights and biases
        self.U, self.b, self.W, self.V, self.c = self.initialize_RNN()

        # Store loss values
        self.training_loss = []

        # Define optimizer
        self.optimizer = torch.optim.Adam([self.U, self.b, self.W, self.V, self.c], lr=1e-3)

    # Initialize network weights and biases using Xavier initialization
    def initialize_RNN(self):
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
            out = xavier_stddev*torch.randn(in_dim, out_dim).type(self.dtype)
            out.requires_grad = True
            return out

        U = xavier_init(size=[self.X_dim, self.hidden_dim])
        b = torch.zeros(1,self.hidden_dim).type(self.dtype)
        b.requires_grad = True

        W = torch.eye(self.hidden_dim).type(self.dtype)
        W.requires_grad = True

        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = torch.zeros(1,self.Y_dim).type(self.dtype)
        c.requires_grad = True

        return U, b, W, V, c

    # Evaluates the forward pass
    def forward_pass(self, X):
        H = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype)
        for i in range(0, self.lags):
            H = torch.tanh(torch.matmul(H,self.W) + torch.matmul(X[i,:,:],self.U) + self.b)
        Y = torch.matmul(H,self.V) + self.c
        return Y

    # Computes the mean square error loss
    def compute_loss(self, X, Y):
        loss = torch.mean((Y - self.forward_pass(X))**2)
        return loss

    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, y, N_batch):
        N = X.shape[1]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[:,idx,:]
        y_batch = y[idx,:]
        return X_batch, y_batch

    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)

            loss = self.compute_loss(X_batch, Y_batch)

            # Store loss value
            self.training_loss.append(loss)

            # Backward pass
            loss.backward()

            # update parameters
            self.optimizer.step()

            # Reset gradients for next step
            self.optimizer.zero_grad()

            # Print
            if (it+1) % 100 == 0:
                Y = self.Y.cpu().numpy()
                Y_pred = self.one_step_ahead(self.X[:,0,:].cpu().numpy().reshape(self.lags,1,-1), self.samples)
                l2e = np.sum(self.l2_norm_error(Y,Y_pred))
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Train loss: %.4e, Normalized L2 norm error: %.4f, Time: %.2f' %
                      (it+1, loss.cpu().data.numpy(), l2e, elapsed))
                start_time = timeit.default_timer()

    # Evaluates predictions at test points
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star

    # One-step-ahead predictions from starting point
    def one_step_ahead(self, startpt, n_predictions):
        Y_pred = np.zeros((n_predictions,2))
        nlags = self.lags
        data = startpt
        for i in range(n_predictions):
            newpt = self.predict(data)
            data[0:nlags-1,:,:] = data[1:nlags,:,:]
            data[nlags-1,:,:] = newpt
            #append generated point to data
            Y_pred[i,:] = data[-1,:,:]
        return Y_pred

    def l2_norm_error(self, X, Y):
        return np.sqrt(np.mean((X - Y)**2, axis = 0))

# LSTM
class LSTM:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):

        self.dtype = torch.FloatTensor
        # X has the form nlags x samples x nfeatures
        # Y has the form samples x nfeatures

        # Define PyTorch variables
        self.X = torch.from_numpy(X).type(self.dtype)
        self.Y = torch.from_numpy(Y).type(self.dtype)
        self.X.requires_grad = False
        self.Y.requires_grad = False

        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]
        self.samples = X.shape[1]

        # LSTM update rule
        # i = sig(HW_i + XU_i + b_i)    -> external input gate
        # f = sig(HW_f + XU_f + b_f)    -> forget gate
        # C = tanh(HW_S + XU_S + b_S)   -> cell state
        # S = f.S + i.C                 -> forget, retain
        # o = sig(HW_o + XU_o + b_o)    -> output gate
        # H = o.tanh(S)                 -> output state
        # Y = HV + c                    -> output vector
        # Parameters: U_i, U_f, U_S, U_o, b_i, b_f, b_S, b_o, W_i, W_f, W_S, W_o, V, c

        # Initialize network weights and biases
        self.U_i, self.U_f, self.U_S, self.U_o, \
        self.b_i, self.b_f, self.b_S, self.b_o, \
        self.W_i, self.W_f, self.W_S, self.W_o, self.V, self.c = self.initialize_LSTM()

        # Store loss values
        self.training_loss = []

        # Define optimizer
        self.optimizer = torch.optim.Adam([self.U_i, self.U_f, self.U_S, self.U_o,\
                                           self.b_i, self.b_f, self.b_S, self.b_o,\
                                           self.W_i, self.W_f, self.W_S, self.W_o, self.V, self.c], lr=1e-3)

    # Initialize network weights and biases using Xavier initialization
    def initialize_LSTM(self):
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
            out = xavier_stddev*torch.randn(in_dim, out_dim).type(self.dtype)
            out.requires_grad = True
            return out

        U_i = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_i = torch.zeros(1,self.hidden_dim).type(self.dtype)
        b_i.requires_grad = True
        W_i = torch.eye(self.hidden_dim).type(self.dtype)
        W_i.requires_grad = True

        U_f = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_f = torch.zeros(1,self.hidden_dim).type(self.dtype)
        b_f.requires_grad = True
        W_f = torch.eye(self.hidden_dim).type(self.dtype)
        W_f.requires_grad = True

        U_S = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_S = torch.zeros(1,self.hidden_dim).type(self.dtype)
        b_S.requires_grad = True
        W_S = torch.eye(self.hidden_dim).type(self.dtype)
        W_S.requires_grad = True

        U_o = xavier_init(size=[self.X_dim, self.hidden_dim])
        b_o = torch.zeros(1,self.hidden_dim).type(self.dtype)
        b_o.requires_grad = True
        W_o = torch.eye(self.hidden_dim).type(self.dtype)
        W_o.requires_grad = True

        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = torch.zeros(1,self.Y_dim).type(self.dtype)
        c.requires_grad = True

        return U_i, U_f, U_S, U_o, b_i, b_f, b_S, b_o, W_i, W_f, W_S, W_o, V, c

    # Evaluates the forward pass
    def forward_pass(self, X):
        H = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype)
        S = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype)
        # i = sig(HW_i + XU_i + b_i)    -> input gate
        # f = sig(HW_f + XU_f + b_f)    -> forget gate
        # g = tanh(HW_S + XU_S + b_S)   -> gate gate
        # S = f.S + i.g                 -> cell state
        # o = sig(HW_o + XU_o + b_o)    -> output gate
        # H = o.tanh(S)                 -> output state
        # Y = HV + c                    -> output vector
        for idx in range(0, self.lags):
            i = torch.sigmoid(torch.matmul(H,self.W_i) + torch.matmul(X[idx,:,:],self.U_i) + self.b_i)
            f = torch.sigmoid(torch.matmul(H,self.W_f) + torch.matmul(X[idx,:,:],self.U_f) + self.b_f)
            o = torch.sigmoid(torch.matmul(H,self.W_o) + torch.matmul(X[idx,:,:],self.U_o) + self.b_o)
            g = torch.tanh(torch.matmul(H,self.W_S) + torch.matmul(X[idx,:,:],self.U_S) + self.b_S)
            S = f * S + i * g
            H = o * torch.tanh(S)
        Y = torch.matmul(H,self.V) + self.c
        return Y

    # Computes the mean square error loss
    def compute_loss(self, X, Y):
        loss = torch.mean((Y - self.forward_pass(X))**2)
        return loss

    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, y, N_batch):
        N = X.shape[1]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[:,idx,:]
        y_batch = y[idx,:]
        return X_batch, y_batch

    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)

            loss = self.compute_loss(X_batch, Y_batch)

            # Store loss value
            self.training_loss.append(loss)

            # Backward pass
            loss.backward()

            # update parameters
            self.optimizer.step()

            # Reset gradients for next step
            self.optimizer.zero_grad()

            # Print
            if (it+1) % 100 == 0:
                Y = self.Y.cpu().numpy()
                Y_pred = self.one_step_ahead(self.X[:,0,:].cpu().numpy().reshape(self.lags,1,-1), self.samples)
                l2e = np.sum(self.l2_norm_error(Y,Y_pred))
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Train loss: %.4e, Normalized L2 norm error: %.4f, Time: %.2f' %
                      (it+1, loss.cpu().data.numpy(), l2e, elapsed))
                start_time = timeit.default_timer()

    # Evaluates predictions at points
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star

    # One-step-ahead predictions from starting point
    def one_step_ahead(self, startpt, n_predictions):
        Y_pred = np.zeros((n_predictions,2))
        nlags = self.lags
        data = startpt
        for i in range(n_predictions):
            newpt = self.predict(data)
            data[0:nlags-1,:,:] = data[1:nlags,:,:]
            data[nlags-1,:,:] = newpt
            #append generated point to data
            Y_pred[i,:] = data[-1,:,:]
        return Y_pred

    def l2_norm_error(self, X, Y):
        return np.sqrt(np.mean((X - Y)**2, axis = 0))
