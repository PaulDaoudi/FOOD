import glob
import os
from collections import defaultdict, OrderedDict
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn


def flatten_completed_trajs(completed_trajs):
    new_completed_trajs = []
    for trajs in completed_trajs:
        nb_trajs = trajs['states'][0].shape[0]
        trajs_list = [{
            'states': [],
            'actions': [],
            'pretanh_actions': [],
        } for _ in range(nb_trajs)]
        for i in range(len(trajs['states'])):
            for j in range(nb_trajs):
                trajs_list[j]['states'].append(trajs['states'][i][j])
                trajs_list[j]['actions'].append(trajs['actions'][i][j])
                trajs_list[j]['pretanh_actions'].append(trajs['pretanh_actions'][i][j])
        for j in range(nb_trajs):
            new_completed_trajs.append(trajs_list[j])
    return new_completed_trajs


class OrderedDefaultDict(OrderedDict):
    def __missing__(self, key):
        self[key] = defaultdict(list)
        return self[key]


class FixSizeOrderedDict(OrderedDefaultDict):
    def __init__(self, *args, max=100, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDefaultDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)


def log_sum_exp(value, dim, keepdim):
    """Numerically stable implementation of the operation:
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=keepdim))


def obs_batch_normalize(obs_tnsr, update_rms, rms_obj):
    """
    Use this function for a batch of 1-D tensors only
    """
    obs_tnsr_np = obs_tnsr.numpy()
    if update_rms:
        rms_obj.update(obs_tnsr_np)

    obs_normalized_np = np.clip((obs_tnsr_np - rms_obj.mean) / np.sqrt(rms_obj.var + 1e-8), -10., 10.)
    obs_normalized_tnsr = torch.FloatTensor(obs_normalized_np).view(obs_tnsr.size()).to(obs_tnsr.device)
    return obs_normalized_tnsr


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class RunningMeanStd:
    """
    https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / (count + batch_count)
    new_var = M2 / (count + batch_count)
    new_count = batch_count + count

    return new_mean, new_var, new_count


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
