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

from unit import * 
from tool import * 

#DIF-Net    
class JointNet_AdaptiveFusion(nn.Module):
    def __init__(
        self,
        num_classes_mod=12,
        num_classes_doa=121,
        embed_dim=64,
        dropout_rate=0.3,
        compressed_len=64,
        reduced_dim=32
    ):
        super().__init__()

        # ========= MOD 分支 =========
        self.mod_conv = nn.Sequential(
            nn.Conv2d(2, 32, (3,1), padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(32, 64, (3,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(64, 128, (3,1), padding=(1,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),
        )

        self.mod_proj = nn.Conv2d(128, embed_dim, 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=2,
            dim_feedforward=128,
            dropout=dropout_rate,
            batch_first=True
        )
        self.mod_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ========= DOA 分支 =========
        self.doa_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ========= 新增：自适应序列压缩（消除长度不对齐） =========
        self.mod_len_compress = nn.AdaptiveAvgPool1d(compressed_len)
        self.doa_len_compress = nn.AdaptiveAvgPool1d(compressed_len)

        # ========= 新增：通道线性降维（减少计算量） =========
        self.mod_channel_reduce = nn.Conv1d(embed_dim, reduced_dim, 1)
        self.doa_channel_reduce = nn.Conv1d(embed_dim, reduced_dim, 1)

        # ========= 跨模态自注意力交互层 =========
        cross_attn_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim,
            nhead=2,
            dim_feedforward=128,
            dropout=dropout_rate,
            batch_first=True
        )
        self.cross_interact = nn.TransformerEncoder(cross_attn_layer, num_layers=2)

        # ========= 分类头（输入维度变为 reduced_dim） =========
        self.mod_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(reduced_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes_mod)
        )

        self.doa_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(reduced_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes_doa)
        )

        # self.doa_head = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.Linear(reduced_dim,128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(128,num_classes_doa),
        #     nn.Sigmoid()
        #     )

    def forward(self, mod_input, doa_input):
        """
        输入：
            mod_input : (B, 2, T, 1)  调制信号输入
            doa_input : (B, 2, H, W)  DOA空间谱输入
        输出：
            mod_out   : (B, num_classes_mod)
            doa_out   : (B, num_classes_doa)
        """
        # ===== MOD 分支 =====
        x_mod = self.mod_conv(mod_input)                # (B, 128, T', 1)
        x_mod = self.mod_proj(x_mod)                    # (B, embed_dim, T', 1)
        x_mod = x_mod.squeeze(-1)                       # (B, embed_dim, T')
        x_mod = x_mod.permute(0, 2, 1)                  # (B, T', embed_dim)
        x_mod = self.mod_transformer(x_mod)             # (B, T', embed_dim)
        x_mod = x_mod.permute(0, 2, 1)                  # (B, embed_dim, T')

        # ===== DOA 分支 =====
        x_doa = self.doa_conv(doa_input)                # (B, embed_dim, H', W')
        x_doa = x_doa.view(x_doa.size(0), x_doa.size(1), -1)  # (B, embed_dim, T_doa)

        # ===== 自适应序列压缩（消除长度差异） =====
        x_mod = self.mod_len_compress(x_mod)            # (B, embed_dim, compressed_len)
        x_doa = self.doa_len_compress(x_doa)            # (B, embed_dim, compressed_len)

        # ===== 通道降维 =====
        x_mod = self.mod_channel_reduce(x_mod)          # (B, reduced_dim, compressed_len)
        x_doa = self.doa_channel_reduce(x_doa)          # (B, reduced_dim, compressed_len)

        # ===== 转换为 Transformer 输入格式 (B, seq_len, dim) =====
        x_mod = x_mod.permute(0, 2, 1)                  # (B, compressed_len, reduced_dim)
        x_doa = x_doa.permute(0, 2, 1)                  # (B, compressed_len, reduced_dim)

        # ===== 跨模态拼接 + 自注意力交互 =====
        combined = torch.cat([x_mod, x_doa], dim=1)     # (B, 2*compressed_len, reduced_dim)
        x = self.cross_interact(combined)               # (B, 2*compressed_len, reduced_dim)

        # ===== 恢复为 (B, C, L) 格式送入分类头 =====
        x = x.permute(0, 2, 1)                          # (B, reduced_dim, 2*compressed_len)

        # ===== 分类 =====
        mod_out = self.mod_head(x)                      # (B, num_classes_mod)
        doa_out = self.doa_head(x)                      # (B, num_classes_doa)

        return mod_out, doa_out