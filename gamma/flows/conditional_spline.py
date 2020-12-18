"""
Adapted from https://github.com/karpathy/pytorch-normalizing-flows
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .nets import ConditionalMLP

from .spline import unconstrained_RQS

class ConditionalNSF(nn.Module):
    """ Conditional version of neural spline flow coupling layer, [Durkan et al. 2019] """

    def __init__(self, dim, K=5, B=3, hidden_dims=[8, 8], condition_dims={'x': 2}, base_network=ConditionalMLP):
        super().__init__()
        self.dim = dim
        self.split_1 = dim // 2
        self.split_2 = dim - dim // 2
        self.K = K
        self.B = B
        self.f1 = base_network(self.split_1, condition_dims, hidden_dims, (3 * K - 1) * self.split_2)
        self.f2 = base_network(self.split_2, condition_dims, hidden_dims, (3 * K - 1) * self.split_1)

    @property
    def requires_condition(self):
        return True
    
    def forward(self, x, condition_dict):
        batch_size = x.shape[0]
        log_det = torch.zeros(x.shape[0])
        lower, upper = x[:, :self.split_1], x[:, self.split_1:]
        out = self.f1(lower, condition_dict).reshape(batch_size, self.split_2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f2(upper, condition_dict).reshape(batch_size, self.split_1, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    def backward(self, z, condition_dict):
        batch_size = z.shape[0]
        log_det = torch.zeros(z.shape[0])
        lower, upper = z[:, :self.split_1], z[:, self.split_1:]
        out = self.f2(upper, condition_dict).reshape(batch_size, self.split_1, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f1(lower, condition_dict).reshape(batch_size, self.split_2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse = True, tail_bound = self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det



