import random

import numpy
import torch


def set_seed_for_all(seed=3305):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    numpy.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.benchmark = False
    torch.backends.deterministic = True
