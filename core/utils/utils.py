import os
import argparse
import datetime
import numpy as np
import _pickle as pickle

import torch


class SmoothCrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def track_bn_stats(model, track_stats=True):
    """
    If track_stats=False, do not update BN running mean and variance and vice versa.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track_stats


def set_bn_momentum(model, momentum=1):
    """
    Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


def str2bool(v):
    """
    Parse boolean using argument parser.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(x):
    """
    Parse float and fractions using argument parser.
    """
    if '/' in x:
        n, d = x.split('/')
        return float(n)/float(d)
    else:
        try:
            return float(x)
        except:
            raise argparse.ArgumentTypeError('Fraction or float value expected.')


def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def seed(seed=1):
    """
    Seed for PyTorch reproducibility.
    Arguments:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def unpickle_data(filename, mode='rb'):
    """
    Read data from pickled file.
    Arguments:
        filename (str): path to the pickled file.
        mode (str): read mode.
    """
    with open(filename, mode) as pkfile:
        data = pickle.load(pkfile)
    return data


def pickle_data(data, filename, mode='wb'):
    """
    Write data to pickled file.
    Arguments:
        data (Any): data to be written.
        filename (str): path to the pickled file.
        mode (str): write mode.
    """
    with open(filename, mode) as pkfile:
         pickle.dump(data, pkfile)


class NumpyToTensor(object):
    """
    Transforms a numpy.ndarray to torch.Tensor.
    """
    def __call__(self, sample): 
        return torch.from_numpy(sample)