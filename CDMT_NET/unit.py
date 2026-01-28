import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader ,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset
from torchsummary import summary
import collections
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 其他依赖
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm
import h5py
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import math
import copy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset, random_split

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_criterion = nn.BCELoss()
        self.doa_criterion = nn.BCELoss()

    def forward(self, mod_pred, doa_pred, mod_target, doa_target, weight):
        mod_loss = self.mod_criterion(mod_pred, mod_target)
        doa_loss = self.doa_criterion(doa_pred, doa_target)

        # 动态权重调整
        total_loss = mod_loss + weight*doa_loss  # 可以根据任务重要性调整权重
        return total_loss, mod_loss, doa_loss


criterion = MultiTaskLoss()

def top_k_binarize(output, k=2):
    batch_size, num_classes = output.shape
    result = torch.zeros_like(output, dtype=torch.float32)
    topk_vals, topk_indices = torch.topk(output, k, dim=1)
    for i in range(batch_size):
        result[i, topk_indices[i]] = 1.0
    return result

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 初始化卷积层权重
        for m in self.shared_mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块，支持多维度kernel_size"""
    def __init__(self, kernel_size=7):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.padding = tuple((k - 1) // 2 for k in self.kernel_size)
        
        self.conv = nn.Conv2d(2, 1, kernel_size=self.kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class ResidualBlockWithCBAM(nn.Module):
    """优化的残差块"""
    def __init__(self, in_channels, out_channels=None, filter_size=(3, 3), 
                 cbam_reduction=4, cbam_kernel_size=7):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=filter_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.channel_att = ChannelAttention(out_channels, cbam_reduction)
        self.spatial_att = SpatialAttention(cbam_kernel_size)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化权重
        for m in self.main_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if not isinstance(self.shortcut, nn.Identity):
            for m in self.shortcut.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        shortcut = self.shortcut(x)
        main_out = self.main_branch(x)
        
        main_out = main_out * self.channel_att(main_out)
        main_out = main_out * self.spatial_att(main_out)
        
        return self.relu(main_out + shortcut)


# ---------------- RMSNorm ---------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
        return self.scale * x / (norm + self.eps)


# ---------------- SwiGLU FFN ---------------- #
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)  # 分成两部分
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj, x_gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x_gate) * x_proj)


# ---------------- RoPE（旋转位置编码） ---------------- #
def apply_rotary_emb(q, k, seq_len, dim):
    """
    q, k: [B, L, D] 或 [B, L, H, D]
    dim: q.shape[-1] // 2   # 只对一半维度应用RoPE
    """
    device = q.device
    half_dim = dim // 2
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # [L, 1]
    freqs = torch.exp(
        -torch.arange(0, half_dim, device=device).float() * (math.log(10000.0) / half_dim)
    )  # [half_dim]
    angles = position * freqs.unsqueeze(0)  # [L, half_dim]

    sin, cos = angles.sin(), angles.cos()  # [L, half_dim]
    # [B, L, half_dim]
    sin, cos = map(lambda t: t.unsqueeze(0).expand(q.shape[0], -1, -1), (sin, cos))

    def rotate(x):
        # x: [B, L, half_dim]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).reshape_as(x)

    # 切分
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    # RoPE
    q1_rot = q1 * cos + rotate(q1) * sin
    k1_rot = k1 * cos + rotate(k1) * sin

    # 拼回
    q_out = torch.cat([q1_rot, q2], dim=-1)
    k_out = torch.cat([k1_rot, k2], dim=-1)
    return q_out, k_out

# ---------------- DropPath ---------------- #
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.empty(x.shape[0], 1, 1, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob

# ---------------- TransformerBlockV2 ---------------- #
class TransformerBlockV2(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=128, dropout=0.1, drop_path=0.1, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RMSNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                RMSNorm(dim),
                SwiGLU(dim, mlp_dim, dropout),
                DropPath(drop_path)
            ]) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape
        for norm1, attn, norm2, ffn, droppath in self.layers:
            # 归一化
            q = k = norm1(x)
            # RoPE
            q, k = apply_rotary_emb(q, k, seq_len, q.shape[-1])
            # 注意力
            attn_out, _ = attn(q, k, x, attn_mask=mask)
            x = x + droppath(attn_out)

            # FFN + 残差
            x = x + droppath(ffn(norm2(x)))
        return self.norm(x)
