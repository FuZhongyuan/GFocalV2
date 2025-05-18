import os
import jittor as jt
import math
import random
import numpy as np
from copy import deepcopy
from patches.torch_adapter import ModelEMA, reduce_sum


def rand_seed(seed=888):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    jt.set_seed(seed)


def convert_weights(weights):
    ret_weights = dict()
    for name, val in weights.items():
        if 'module.' in name:
            name = name.replace('module.', '')
        ret_weights[name] = val
    return ret_weights


def is_parallel(model):
    # Jittor不需要像PyTorch那样显式处理并行
    return False


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


# ModelEMA已在torch_adapter中实现


class AverageLogger(object):
    def __init__(self):
        self.data = 0.
        self.count = 0.

    def update(self, data, count=None):
        self.data += data
        if count is not None:
            self.count += count
        else:
            self.count += 1

    def avg(self):
        return self.data / self.count

    def sum(self):
        return self.data

    def reset(self):
        self.data = 0.
        self.count = 0. 