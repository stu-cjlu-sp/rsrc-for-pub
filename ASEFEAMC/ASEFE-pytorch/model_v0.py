# import math
# from functools import partial
# from typing import Optional, Callable
# from torch import Tensor
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from einops import rearrange, repeat
# from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
# from mamba_ssm import Mamba
# from functools import partial
from torchinfo import summary

# try:
#     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
#     print("[INFO] Using fused selective_scan_fn from mamba_ssm")
# except ImportError as e:
#     raise RuntimeError("❌ Failed to import fused selective_scan_fn from mamba_ssm. You're using slow fallback.")



##############   SEFEFeatureExtractor   ##############
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction_ratio=1):
        super(SEBlock1D, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, C, L)
        b, c, l = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class SEFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEFEBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.combine = nn.Sequential(
            nn.Conv1d(in_channels + 3 * out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.se = SEBlock1D(out_channels, reduction_ratio=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L)

        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)

        out = torch.cat([x, c1, c2, c3], dim=1)  # (B, in_channels + 3*out_channels, L)
        out = self.combine(out)
        out = self.se(out)

        return out.permute(0, 2, 1)  # (B, L, C)

def position_encoding(seq_len, dim):
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    dim_idx = torch.arange(dim, dtype=torch.float).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (dim_idx // 2)) / dim)
    angle_rads = pos * angle_rates

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return angle_rads.unsqueeze(0)  # (1, seq_len, dim)

class SEFEFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2, seq_len=128, num_classes=11):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.sefe_blocks = nn.Sequential(
            SEFEBlock(64, 32), SEFEBlock(32, 32), SEFEBlock(32, 32),
            SEFEBlock(32, 32), SEFEBlock(32, 32), SEFEBlock(32, 32)
        )
        self.pos_emb = nn.Parameter(position_encoding(seq_len, 32), requires_grad=False)
        self.transformer = TransformerBlock(dim=32, num_heads=2, ff_dim=32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):  # (B, 2, 128)
        x = x.permute(0, 2, 1)  # -> (B, 128, 2)
        x, _ = self.bilstm(x)   # -> (B, 128, 64)
        x = self.dropout(x)
        x = self.sefe_blocks(x)  # -> (B, 128, 32)
        x = x + self.pos_emb
        x = self.transformer(x)  # -> (B, 128, 32)
        x = x.transpose(1, 2)    # -> (B, 32, 128)
        x = self.pool(x).squeeze(-1)  # -> (B, 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



if __name__ == '__main__':
    ##### Test VSSM Flatten  ####
    # medmamba_flatten = VSSM_Flatten(in_chans=1, depths=[2, 2, 4, 2], dims=[128, 128, 128, 128], num_classes=11,d_state=16, patch_size=4).to("cuda")
    # data = torch.randn(1, 1, 224, 224).to("cuda")
    # print(medmamba_flatten(data).shape)
    # summary(medmamba_flatten, input_size=(1, 1, 224, 224))

    ##### Test SEFEFeatureExtractor ####
    SEFE_feature_extractor = SEFEFeatureExtractor(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    data = torch.randn(1, 2, 128).to("cuda")
    print(SEFE_feature_extractor(data).shape)

    summary(SEFE_feature_extractor, input_size=(1, 2, 128))  # (batch, channel, seq_len)

    ##### Test CNNModel ####
    # cnn = CNNModel().to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(cnn(data).shape)
    #
    # summary(cnn, input_size=(1, 2, 128))  # (batch, channel, seq_len)

    ##### Test SEFEMambaFeatureExtractor ####
    # SEFE_mamba_feature_extractor = SEFEMambaFeatureExtractor(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(SEFE_mamba_feature_extractor(data).shape)
    #
    # summary(SEFE_mamba_feature_extractor, input_size=(1, 2, 128))  # (batch, channel, seq_len)


    ##### Test SEFE+Mamba+LocalEnhancer ####
    # SEFE_mamba_localEnhancer = SEFEMambaLocal(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(SEFE_mamba_localEnhancer(data).shape)
    #
    # summary(SEFE_mamba_localEnhancer, input_size=(1, 2, 128))  # (batch, channel, seq_len)

    ##### Test SEFE_PeriodicConv_MambaLocal ####
    # SEFE_periodicconv_mamba_local = SEFE_PeriodicConv_MambaLocal(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(SEFE_periodicconv_mamba_local(data).shape)
    # summary(SEFE_periodicconv_mamba_local, input_size=(1, 2, 128))  # (batch, channel, seq_len)

    ##### Test SEFEMambaWithPeriod ####
    ##### (Mamba, LearnablePeriod Estimator, SEFE, PeriodicConv, Mamba, LocalEnhancer) ######
    # learnablePeriod_SEFE_mamba = SEFEMambaWithPeriod(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(learnablePeriod_SEFE_mamba(data).shape)
    # summary(learnablePeriod_SEFE_mamba, input_size=(1, 2, 128))  # (batch, channel, seq_len)

    ##### Test SEFE_PeriodAwareMamba_MambaLocal ####

    # SEFE_period_aware_mamba_mambaLocal = SEFE_PeriodAwareMamba_MambaLocal(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(SEFE_period_aware_mamba_mambaLocal(data).shape)
    # summary(SEFE_period_aware_mamba_mambaLocal, input_size=(1, 2, 128))  # (batch, channel, seq_len)


    ########## 对比模型  ###############
    ##### Test 1. MambaSEFEMamba ############
    # mamba_SEFE_mamba = MambaSEFEMamba(input_dim=2, seq_len=128, num_classes=11).to("cuda")
    # data = torch.randn(1, 2, 128).to("cuda")
    # print(mamba_SEFE_mamba(data).shape)
    #
    # summary(mamba_SEFE_mamba, input_size=(1, 2, 128))  # (batch, channel, seq_len)
