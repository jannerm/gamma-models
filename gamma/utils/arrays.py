import numpy as np
import torch

DEVICE = 'cpu'
DTYPE = torch.float

def set_device(device):
	global DEVICE
	DEVICE = device

	if 'cuda' in device:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)
	else:
		torch.set_default_tensor_type(torch.FloatTensor)

def to_np(x):
	return x.detach().cpu().numpy()

def to_torch(x, **kwargs):
	if torch.is_tensor(x):
		return x
	else:
		return torch.tensor(x, device=DEVICE, dtype=DTYPE, **kwargs)

def dict_to_torch(d):
	return {
		key: to_torch(val)
		for key, val in d.items()
	}