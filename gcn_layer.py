import numpy as np
from scipy.sparse.linalg import eigs
import torch
import torch.nn as nn
import math


def Scaled_Laplacian(W):
    W = W.astype(float)
    n = np.shape(W)[0]
    d = []
    L = -W
    for i in range(n):
        d.append(np.sum(W[i, :]))
        L[i, i] = d[i]
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.matrix(2 * L / lambda_max - np.identity(n))

def Cheb_Poly(L, Ks):
    assert L.shape[0] == L.shape[1]
    n = L.shape[0]
    L0 = np.matrix(np.identity(n))
    L1 = np.matrix(np.copy(L))
    L_list = [np.copy(L0), np.copy(L1)]
    for i in range(1, Ks):
        Ln = np.matrix(2 * L * L1 - L0)
        L_list.append(np.copy(Ln))
        L0 = np.matrix(np.copy(L1))
        L1 = np.matrix(np.copy(Ln))
    # L_lsit (Ks, n*n), Lk (n, Ks*n)
    return np.concatenate(L_list, axis=-1)

def First_Approx(W):
    n = W.shape[0]
    A = W + np.identity(n, dtype=np.float32)
    d = []
    for i in range(n):
        d.append(np.sum(A[i, :]))
    sinvD = np.sqrt(np.matrix(np.diag(d)).I)
    return np.identity(n, dtype=np.float32) + sinvD * A * sinvD

def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

class GConv(nn.Module):
    def __init__(self, in_dim, out_dim, order, bias=True, init='xavier', cuda=True):
        super(GConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.order = order
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.weight = torch.nn.parameter.Parameter(Tensor(self.order*in_dim, out_dim))
        if bias:
            self.bias = torch.nn.parameter.Parameter(Tensor(out_dim))
        if init == 'uniform':
            self.reset_parameters_uniform()
        elif init == 'xavier':
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, supports):

        num_nodes = input.shape[1]
        assert num_nodes == supports.shape[0]
        output = torch.einsum('ij, bik -> bjk', supports, input)
        output = output.reshape(output.shape[0], num_nodes, -1)
        output = torch.matmul(output, self.weight)  # [B, N, out_dim]
        if self.bias is not None:
            return output + self.bias
        else:
            return output
