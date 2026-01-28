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

class CDMT_Net(nn.Module):
    def __init__(self, num_classes_mod=12, num_classes_doa=81, embed_dim=64, dropout_rate=0.3):
        super().__init__()
        
        # ---------------- MOD分支（时域特征） ----------------
        self.mod_initial = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3,1), padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.mod_residuals = nn.Sequential(
            ResidualBlockWithCBAM(32,64,filter_size=(3,1),cbam_kernel_size=(3,1)),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            ResidualBlockWithCBAM(64,64,filter_size=(3,1),cbam_kernel_size=(3,1)),
            ResidualBlockWithCBAM(64,64,filter_size=(3,3),cbam_kernel_size=3)
        )
        self.mod_proj_to_embed = nn.Conv2d(64, embed_dim, kernel_size=(1,1), bias=False)
        self.mod_transformer = TransformerBlockV2(dim=embed_dim, heads=2, mlp_dim=128, dropout=dropout_rate, num_layers=2)
        
        # ---------------- DOA分支（空间特征） ----------------
        self.doa_initial = nn.Sequential(
            nn.Conv2d(2,32,kernel_size=(3,3), padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.doa_residuals = nn.Sequential(
            ResidualBlockWithCBAM(32,64,filter_size=(3,3),cbam_kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ResidualBlockWithCBAM(64,64,filter_size=(3,3),cbam_kernel_size=3),
            ResidualBlockWithCBAM(64,64,filter_size=(3,3),cbam_kernel_size=3)
        )
        self.doa_proj_to_embed = nn.Conv2d(64, embed_dim, kernel_size=(1,1), bias=False)
        
        # ---------------- 高层共享部分 ----------------
        self.shared_dropout = nn.Dropout(dropout_rate)
        
        # ---------------- 任务头 ----------------
        self.mod_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes_mod)
        )
        self.doa_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes_doa)
        )
        
        # 初始化BN层
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, mod_input=None, doa_input=None, task='both', w_mod=0.5, w_doa=0.5):
        mod_out = doa_out = result_mod = result_doa = None
        x_mod_embed = x_doa_embed = None
        
        # ---------------- MOD分支 ----------------
        if task in ['mod','both'] and mod_input is not None:
            x_mod = mod_input.squeeze(1)
            x_mod = self.mod_initial(x_mod)
            x_mod = self.mod_residuals(x_mod)
            x_mod = self.mod_proj_to_embed(x_mod)
            B,C,T,N = x_mod.shape
            x_mod = x_mod.permute(0,2,3,1).contiguous().view(B,T*N,C)
            x_mod = self.mod_transformer(x_mod)
            x_mod_embed = self.shared_dropout(x_mod.permute(0,2,1))
        
        # ---------------- DOA分支 ----------------
        if task in ['doa','both'] and doa_input is not None:
            x_doa = doa_input.squeeze(1)
            x_doa = self.doa_initial(x_doa)
            x_doa = self.doa_residuals(x_doa)
            x_doa = self.doa_proj_to_embed(x_doa)
            B,C,H,W = x_doa.shape
            x_doa_embed = self.shared_dropout(x_doa.view(B,C,H*W))
        
        # ---------------- 高层加权融合 ----------------
        if x_mod_embed is not None and x_doa_embed is not None:
            min_len = min(x_mod_embed.shape[2], x_doa_embed.shape[2])
            x_mod_trunc = x_mod_embed[:,:,:min_len]
            x_doa_trunc = x_doa_embed[:,:,:min_len]
            x_shared = w_mod * x_mod_trunc + w_doa * x_doa_trunc
        elif x_mod_embed is not None:
            x_shared = x_mod_embed
        elif x_doa_embed is not None:
            x_shared = x_doa_embed
        else:
            return None,None,None,None
        
        # ---------------- 任务头 ----------------
        if task in ['mod','both'] and x_mod_embed is not None:
            mod_out = self.mod_head(x_shared)
            result_mod = torch.argmax(F.softmax(mod_out, dim=1), dim=1)
        
        if task in ['doa','both'] and x_doa_embed is not None:
            doa_out = self.doa_head(x_shared)
            result_doa = torch.argmax(F.softmax(doa_out, dim=1), dim=1)
        
        return mod_out, doa_out, result_mod, result_doa