import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pdb


class ConditionalNormalizingFlow(nn.Module):

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, condition):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            if hasattr(flow, 'requires_condition') and flow.requires_condition:
                x, ld = flow.forward(x, condition)
            else:
                x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, condition):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            if hasattr(flow, 'requires_condition') and flow.requires_condition:
                z, ld = flow.backward(z, condition)
            else:
                z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

    def transform(self, z, condition):
        xs, _ = self.backward(z, condition)
        x = xs[-1]
        return x

class DeltaFlow(ConditionalNormalizingFlow):

    """
        adds the output of the flow model to condition['s']
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, condition):
        delta = x - condition['s']
        return super().forward(delta, condition)

    def backward(self, z, condition):
        deltas, log_det = super().backward(z, condition)
        deltas[-1] = deltas[-1] + condition['s']
        return deltas, log_det

class ConditionalNormalizingFlowModel(nn.Module):
    
    def __init__(self, prior, flows, delta=False):
        super().__init__()
        self.prior = prior
        if delta:
            self.flow = DeltaFlow(flows)
        else:
            self.flow = ConditionalNormalizingFlow(flows)
    
    def forward(self, x, condition=None):
        zs, log_det = self.flow.forward(x, condition)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, condition=None):
        xs, log_det = self.flow.backward(z, condition)
        return xs, log_det

    def sample(self, num_samples, condition=None):
        z = self.prior.sample((num_samples,))
        x = self.flow.transform(z, condition)
        return x

    def log_prob(self, x, condition=None):
        zs, prior_logprob, log_det = self.forward(x, condition)
        logprob = prior_logprob + log_det
        return logprob

    def grad_logp(self, x, condition=None):
        logp = self.log_prob(x, condition)
        grad_logp = torch.autograd.grad(logp.sum(), x, retain_graph=True, create_graph=True)[0]
        return grad_logp












