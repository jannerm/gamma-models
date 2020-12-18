import torch
import torch.nn as nn
import pdb


def get_activation(params):
	if type(params) == dict:
		name = params['type']
		kwargs = params['kwargs']
	else:
		name = str(params)
		kwargs = {}
	return lambda: getattr(nn, name)(**kwargs)

def flatten(condition_dict):
	keys = sorted(condition_dict)
	vals = [condition_dict[key] for key in keys]
	condition = torch.cat(vals, dim=-1)
	return condition

class MLP(nn.Module):

	def __init__(self, input_dim, hidden_dims, output_dim, activation='ReLU', output_activation='Identity', name='mlp'):
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.name = name
		activation = get_activation(activation)
		output_activation = get_activation(output_activation)

		layers = []
		current = input_dim
		for dim in hidden_dims:
			linear = nn.Linear(current, dim)
			layers.append(linear)
			layers.append(activation())
			current = dim

		layers.append(nn.Linear(current, output_dim))
		layers.append(output_activation())

		self._layers = nn.Sequential(*layers)

	def forward(self, x):
		return self._layers(x)

	@property
	def num_parameters(self):
		parameters = filter(lambda p: p.requires_grad, self.parameters())
		return sum([p.numel() for p in parameters])
	
	def __repr__(self):
		return  '[ {} : {} parameters ] {}'.format(
			self.name, self.num_parameters,
			super().__repr__())


class ConditionalMLP(MLP):

	def __init__(self, input_dim, condition_dims, *args, verbose=False, **kwargs):
		self._condition_dims = condition_dims
		self._condition_keys = sorted(self._condition_dims.keys())

		concat_dim = input_dim + sum(condition_dims.values())
		super(ConditionalMLP, self).__init__(concat_dim, *args, **kwargs)

		if verbose:
			print('[ conditional mlp ] Conditioning keys: {}'.format(self._condition_keys))
			print(self)

	def forward(self, x, condition_dict):
		if condition_dict is None:
			joined = x
		else:
			condition = flatten(condition_dict)
			joined = torch.cat([x, condition], dim=-1)

		out = self._layers(joined)

		return out

	def __repr__(self):
		return  '[ {} : {} parameters | {} ] {}'.format(
			self.name, self.num_parameters,
			self._condition_dims,
			super(MLP, self).__repr__())

