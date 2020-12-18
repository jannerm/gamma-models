import io
import numpy as np
import matplotlib.pyplot as plt

from gamma.td.utils import make_condition

from gamma.utils.arrays import (
    to_torch,
    to_np,
)

def make_prob_fn(model, policy):

    def _fn(states, queries):
        actions = policy(states)
        queries = to_torch(queries)
        condition_dict = make_condition(states, actions)
        logp = model.log_prob(queries, condition_dict)
        prob = logp.exp()
        return to_np(prob)

    return _fn