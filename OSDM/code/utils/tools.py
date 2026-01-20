import os
import random
import numpy as np
import torch
import math
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def pos_enc(pos, model_dim):
    p = torch.arange(pos, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(model_dim, dtype=torch.float32).unsqueeze(0)
    rates = 1 / torch.pow(10000, (2 * (dims // 2)) / model_dim)
    rads = p * rates
    enc = torch.zeros((pos, model_dim), dtype=torch.float32)
    enc[:, 0::2] = torch.sin(rads[:, 0::2])
    enc[:, 1::2] = torch.cos(rads[:, 1::2])
    return enc.unsqueeze(0)


def pad_mask(seq):
    seq = (seq == 0).float()
    return seq.unsqueeze(1).unsqueeze(2)


def warmup_scheduler(optimizer, warm_steps, total_steps, min_factor=1e-5):
    def lr_scale(current_step: int):
        if current_step < warm_steps:
            return float(current_step) / float(max(1, warm_steps))
        return max(
            min_factor,
            0.5 * (1 + math.cos(math.pi * (current_step - warm_steps) / (total_steps - warm_steps)))
        )
    return LambdaLR(optimizer, lr_scale)