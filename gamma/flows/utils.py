import os
import itertools
import torch

from torch.distributions import (
    Uniform,
    TransformedDistribution,
    SigmoidTransform,
)

from .flows import (
    ActNorm,
    Invertible1x1Conv,
)
from .conditional_spline import ConditionalNSF

from .conditional import ConditionalNormalizingFlowModel


def make_conditional_flow(dim=2, hidden_dims=[16, 16], condition_dims={'x': 2}, B=10, K=16, num_layers=3, delta=False):
    ## prior
    uniform = Uniform(torch.zeros(dim), torch.ones(dim))
    logistic = TransformedDistribution(uniform, SigmoidTransform().inv)

    # conditional flow layers
    coupling_layer = ConditionalNSF
    flows = [coupling_layer(
                dim=dim, K=K, B=B, hidden_dims=hidden_dims, condition_dims=condition_dims)
            for _ in range(num_layers)]
    convs = [Invertible1x1Conv(dim=dim) for _ in flows]
    norms = [ActNorm(dim=dim) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))

    # compose the layers into a conditional flow model
    model = ConditionalNormalizingFlowModel(logistic, flows, delta=delta)

    return model

def save_model(path, epoch, model):
    fullpath = os.path.join(path, '{}.pt'.format(epoch))
    state_dict = model.state_dict()
    torch.save(state_dict, fullpath)
    print('Saved state dict to {}'.format(fullpath))

def load_model(path, epoch, model):
    fullpath = os.path.join(path, '{}.pt'.format(epoch))
    state_dict = torch.load(fullpath)
    model.load_state_dict(state_dict, strict=True)
    for flow in model.flow.flows:
        if hasattr(flow, 'data_dep_init_done'):
            setattr(flow, 'data_dep_init_done', True)
            print('{} : Set data_dep_init_done=True'.format(flow))
    print('Loaded state dict from {}'.format(fullpath))

