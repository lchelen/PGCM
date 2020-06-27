import torch
import torch.nn as nn
from gcn_layer import GConv

class GonvST(nn.Module):
    def __init__(self, kt, dim_in, dim_out, order, activation, residual=True):
        super(GonvST, self).__init__()
        self.activation = activation
        self.residual = residual
        self.kt = kt
        self.dim_in = dim_in
        self.dim_out = dim_out
        if activation == 'GLU':
            self.gcn = GConv(in_dim=kt * dim_in, out_dim=2 * dim_out, order=order)
        else:
            self.gcn = GConv(in_dim=kt * dim_in, out_dim=dim_out, order=order)
        if self.activation == 'GLU':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.ReLU()

    def forward(self, x, supports):
        input = x
        assert self.dim_in == x.shape[3]
        if self.dim_in == self.dim_out:
            res_input = input
        elif self.dim_in < self.dim_out:
            device = input.get_device()
            padding = torch.zeros(input.size(0), input.size(1), input.size(2), self.dim_out - self.dim_in).to(device)
            res_input = torch.cat([input, padding], dim=3)
        res_input = res_input[:, self.kt - 1:, :, :]

        T = x.size(1)
        node = x.size(2)
        dim_in = x.size(3)
        x = torch.stack([x[:, i:i + self.kt, :, :] for i in range(0, T - self.kt + 1)], dim=1)
        x = x.reshape(-1, self.kt, node, dim_in)
        #x = torch.cat([x[:, i:i + self.kt, :, :] for i in range(0, T - self.kt + 1)], dim=0)  # (B*, kt, N, F)
        x = x.permute(0, 2, 1, 3).reshape(-1, node, self.kt * dim_in)
        if self.activation == 'GLU':
            out = self.gcn(x, supports).reshape(-1, T - self.kt + 1, node, 2 * self.dim_out)
            out = (out[:, :, :, 0:self.dim_out] + res_input) * self.act(out[:, :, :, self.dim_out:])
        else:
            out = self.gcn(x, supports).reshape(-1, T - self.kt + 1, node, self.dim_out)
            out = self.act(out)
        return out

class GCN2S(nn.Module):
    def __init__(self, supports, args, dropout):
        super(GCN2S, self).__init__()
        self.args = args
        self.patch = 3
        self.act = 'GLU'
        self.node_num = supports.shape[0]
        self.dropout = nn.Dropout(p=dropout)
        self.order = int(supports.shape[1] / self.node_num)
        self.supports = torch.nn.parameter.Parameter(supports, requires_grad=False)
        self.stgcn1 = GonvST(self.patch, dim_in=args.dim, dim_out=32, order = self.order,
                             activation=self.act, residual=True)
        self.stgcn2 = GonvST(self.patch, dim_in=32, dim_out=32, order = self.order,
                             activation=self.act, residual=True)
        self.stgcn3 = GonvST(self.patch, dim_in=32, dim_out=64, order = self.order,
                             activation=self.act, residual=True)
        self.stgcn4 = GonvST(self.patch, dim_in=64, dim_out=64, order = self.order,
                             activation=self.act, residual=True)
        self.stgcn5 = GonvST(self.patch, dim_in=64, dim_out=64, order = self.order,
                             activation=self.act, residual=True)
        #self.stgcn6 = GonvST(self.patch, dim_in=32, dim_out=32, order=self.order,
                             #activation=self.act, residual=True)
        #self.stgcn7 = GonvST(self.patch, dim_in=32, dim_out=32, order=self.order,
                             #activation=self.act, residual=True)
        self.stgcn8 = GonvST(2, dim_in=64, dim_out=64, order = self.order,
                             activation=self.act, residual=True)
        self.conv = nn.Conv2d(64, args.dim, kernel_size=(1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # shape of x: [B, W, N]  -- [B, T, N]
        if len(x.shape) == 3:
            out = x.unsqueeze(dim=3)  # [B, T, N, 1]
        else:
            out = x
        out = self.dropout(self.stgcn1(out, self.supports))
        out = self.dropout(self.stgcn2(out, self.supports))
        out = self.dropout(self.stgcn3(out, self.supports))
        out = self.dropout(self.stgcn4(out, self.supports))
        out = self.dropout(self.stgcn5(out, self.supports))
        #out = self.stgcn6(out, self.supports)
        #out = self.stgcn7(out, self.supports)
        out = self.dropout(self.stgcn8(out, self.supports))
        out = out.permute(0, 3, 1, 2)
        out = self.dropout(self.conv(out))
        out = out.permute(0, 2, 3, 1).squeeze(dim=3)
        return out

