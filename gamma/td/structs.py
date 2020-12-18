import numpy as np
import pickle
import torch

from gamma.utils.arrays import (
    to_torch,
    to_np,
    dict_to_torch,
)

class ReplayPool:

    def __init__(self, loadpath):
        with open(loadpath, 'rb') as f:
            self.fields = pickle.load(f)

        ## ensure that all fields have the same first dimension
        sizes = [len(val) for key, val in self.fields.items()]
        assert len(set(sizes)) == 1
        self.size = sizes[1]

    def __getitem__(self, key):
        return self.fields[key]

    def sample(self, batch_size):
        inds = np.random.randint(
            low=0, high=self.size, size=batch_size)
        batch = {key: val[inds] for key, val in self.fields.items()}
        return dict_to_torch(batch)

class Policy:
    """
        lightweight wrapper around rlkit policy
    """

    def __init__(self, loadpath):
        snapshot = torch.load(loadpath)
        self._policy = snapshot['evaluation/policy']

    def __call__(self, observations):
        observations = to_torch(observations)
        actions = self._policy.get_actions(observations)
        return actions
