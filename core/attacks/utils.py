import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.distributions import laplace
from torch.distributions import uniform
from torch.nn.modules.loss import _Loss


def replicate_input(x):
    """
    Clone the input tensor x.
    """
    return x.detach().clone()


def replicate_input_withgrad(x):
    """
    Clone the input tensor x and set requires_grad=True.
    """
    return x.detach().clone().requires_grad_()


def calc_l2distsq(x, y):
    """
    Calculate L2 distance between tensors x and y.
    """
    d = (x - y)**2
    return d.view(d.shape[0], -1).sum(dim=1)


def clamp(input, min=None, max=None):
    """
    Clamp a tensor by its minimun and maximun values.
    """
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following.
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following.
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    """
    Multpliy a batch of tensors with a float or vector.
    """
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def batch_clamp(float_or_vector, tensor):
    """
    Clamp a batch of tensors.
    """
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _get_norm_batch(x, p):
    """
    Returns the Lp norm of batch x.
    """
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def _thresh_by_magnitude(theta, x):
    """
    Threshold by magnitude.
    """
    return torch.relu(torch.abs(x) - theta) * x.sign()


def clamp_by_pnorm(x, p, r):
    """
    Clamp tensor by its norm.
    """
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def is_float_or_torch_tensor(x):
    """
    Return whether input x is a float or a torch.Tensor.
    """
    return isinstance(x, torch.Tensor) or isinstance(x, float)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    Arguments:
        x (torch.Tensor): tensor containing the gradients on the input.
        p (int): (optional) order of the norm for the normalization (1 or 2).
        small_constant (float): (optional) to avoid dividing by zero.
    Returns: 
        normalized gradients.
    """
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    """
    Randomly initialize the perturbation.
    """
    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss
