import os
import argparse

parser = argparse.ArgumentParser(description="CNN")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="3", help='GPU id')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from timm.layers import trunc_normal_

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
# from collections import defaultdic

# 其他依赖
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import h5py
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import math
import copy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
import torch.fft
from tool import * 
from thop import profile, clever_format

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Conv_Block, self).__init__()
        kernel_size = kernel_size if kernel_size is not None else (1,3)
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1,1,0,0)),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )
    def forward(self, x):
        return self.conv_block(x)

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.dropout = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class TinyMLP(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N, N//4),
            nn.ReLU(inplace=True),
            nn.Linear(N//4, N),
            nn.Tanh()
        )
    def forward(self, x):
        # x shape: (B, C, D, W) -> after permute (B, C, W, D)
        # 这里线性层作用在最后一维 D 上，但原始代码意图是作用在 W 上，已根据输入调整
        # 为确保稳定，保持原逻辑
        x = self.mlp(x)
        return x

class FrequencyDomainDenoisingModule(nn.Module):
    def __init__(self, N, C=2):
        super().__init__()
        self.mlp = TinyMLP(N)

    def forward(self, x):
        x_init = x.clone()
        B, C, W, D = x.shape

        # 1. 组合实部虚部为复数信号
        complex_signal = x[:, 0, :, :] + 1j * x[:, 1, :, :]   # (B, W, D)

        # 2. 在宽度维度上做 FFT
        fft_signal = torch.fft.fft(complex_signal, dim=1)     # (B, W, D)

        # 3. 分离实部虚部，作为双通道输入
        x_fft = torch.stack([fft_signal.real, fft_signal.imag], dim=1)  # (B, 2, W, D)

        # 4. 调整维度顺序，使 MLP 作用于宽度维 (W)
        # 原始: (B,2,W,D) → 将 D 移到第2维，W 移到最后: (B,2,D,W)
        x_mlp = x_fft.permute(0, 1, 3, 2)   # (B, 2, D, W)
        h = self.mlp(x_mlp)                 # (B, 2, D, W)

        # 5. 恢复为 (B, 2, W, D) 顺序
        h = h.permute(0, 1, 3, 2)           # (B, 2, W, D)

        # 6. 频域滤波：逐元素相乘
        filtered_real = h[:, 0, :, :] * x_fft[:, 0, :, :]   # (B, W, D)
        filtered_imag = h[:, 1, :, :] * x_fft[:, 1, :, :]   # (B, W, D)
        filtered_signal = filtered_real + 1j * filtered_imag  # (B, W, D)

        # 7. IFFT 回到时域
        ifft_signal = torch.fft.ifft(filtered_signal, dim=1)   # (B, W, D)

        # 8. 重建输出
        out = torch.zeros_like(x)
        out[:, 0, :, :] = ifft_signal.real
        out[:, 1, :, :] = ifft_signal.imag

        return out + x_init   # 残差连接

class MCDformer(nn.Module):
    def __init__(self, sig_len=768, num_classes=12):
        super().__init__()
        self.sig_len = sig_len
        self.num_classes = num_classes
        self.latent_dim = 256
        self.num_heads = 8
        self.conv_chan_list = [2, 36, 64, 128, 256]

        self.FDDM = FrequencyDomainDenoisingModule(self.sig_len, C=2)

        kernel_size_list = [(1,3), (1,3), (1,3), (1,3)]
        self.Conv_stem = nn.Sequential()
        for t in range(len(self.conv_chan_list)-1):
            self.Conv_stem.add_module(f'conv_stem_{t}',
                                     Conv_Block(self.conv_chan_list[t],
                                               self.conv_chan_list[t+1],
                                               kernel_size_list[t]))
        self.block = Block(dim=self.latent_dim, num_heads=self.num_heads, mlp_ratio=1)
        self.block_conv = ConvLayer(self.conv_chan_list[-1])
        self.block2 = Block(dim=self.latent_dim, num_heads=self.num_heads, mlp_ratio=1)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

    def forward(self, x):
        # 适配输入形状：允许 (B,2,L) 或 (B,2,L,1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)          # (B,2,L,1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        B, C, L, H = x.shape
        # 确保高度维为1（如果大于1，则通过平均压缩，但原设计期望1）
        if H != 1:
            x = x.mean(dim=3, keepdim=True)   # (B,2,L,1)
            H = 1

        x = self.FDDM(x)               # (B,2,L,1)
        x = self.Conv_stem(x)          # (B,256,L,1)
        x = torch.mean(x, dim=2)       # (B,256,L)  压缩高度维
        x = x.permute(0, 2, 1)         # (B,L,256)
        x = self.block(x)              # (B,L,256)
        x = x.permute(0, 2, 1)         # (B,256,L)
        x = self.block_conv(x)         # (B,256,L//2)
        x = x.permute(0, 2, 1)         # (B,L//2,256)
        x = self.block2(x)             # (B,L//2,256)
        x = x[:, -1, :]                # (B,256)
        out = self.classifier(x)       # (B,num_classes)
        return out
        

    
def train_epoch_mod(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for iq, mod_label, _ in loader:   # 假设 loader 返回 (iq, mod, snr)
        iq = iq.to(device)
        mod_label = mod_label.to(device)
        optimizer.zero_grad()
        output = model(iq)
        loss = criterion(output, mod_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * iq.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(mod_label).sum().item()
        total += iq.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def eval_epoch_mod(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for iq, mod_label, _ in loader:
            iq = iq.to(device)
            mod_label = mod_label.to(device)
            output = model(iq)
            loss = criterion(output, mod_label)
            total_loss += loss.item() * iq.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(mod_label).sum().item()
            total += iq.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def test_by_snr_and_mod(model, loader, device, num_classes):
    model.eval()
    from collections import defaultdict
    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)
    mod_correct = defaultdict(int)
    mod_total = defaultdict(int)
    with torch.no_grad():
        for iq, mod_label, snr in loader:
            iq = iq.to(device)
            mod_label = mod_label.to(device)
            snr = snr.item() if torch.is_tensor(snr) else snr
            output = model(iq)
            pred = output.argmax(dim=1)
            # 按 SNR
            snr_correct[snr] += (pred == mod_label).sum().item()
            snr_total[snr] += iq.size(0)
            # 按调制类型
            for m in range(num_classes):
                mask = (mod_label == m)
                if mask.any():
                    mod_correct[m] += (pred[mask] == m).sum().item()
                    mod_total[m] += mask.sum().item()
    print("\n===== Accuracy per SNR =====")
    for snr in sorted(snr_total.keys()):
        acc = 100.0 * snr_correct[snr] / snr_total[snr]
        print(f"SNR {snr:3d} dB : {acc:5.2f}%  ({snr_correct[snr]}/{snr_total[snr]})")
    print("\n===== Accuracy per Modulation =====")
    for m in range(num_classes):
        if mod_total[m] > 0:
            acc = 100.0 * mod_correct[m] / mod_total[m]
            print(f"Mod {m:2d} : {acc:5.2f}%  ({mod_correct[m]}/{mod_total[m]})")

def plot_curves(history, save_dir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

# -------------------- Main 函数 --------------------
def main_mod():
    result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/mod/MCDFormer'
    os.makedirs(result_dir, exist_ok=True)
    file_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/dataset'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 128
    num_epochs = 120
    lr = 1e-3
    weight_decay = 1e-4
    patience = 10

    # 加载数据集（IQ + 调制标签 + SNR）
    dataset = H5MultiTaskDataset(
        folder_path=file_path,
        return_cov=False,
        return_spectrum=False,
        return_angle=False,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'          # 返回 (samples, 2, snapshot) 形状
    )

    # 获取 IQ 序列长度（快拍数）
    iq_shape = dataset.iq_real.shape   # (samples, snapshot)
    seq_len = iq_shape[1]              # 应为 768
    print(f"Detected IQ sequence length: {seq_len}")

    # 按 SNR 划分训练/验证/测试集
    snrs = dataset.snr
    snr_to_indices = defaultdict(list)
    for idx, snr_val in enumerate(snrs):
        snr_to_indices[snr_val].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    print("\nSNR-wise sample counts (train/val/test):")
    for snr_val, indices in snr_to_indices.items():
        n = len(indices)
        if n < 5:
            print(f"SNR {snr_val} has too few samples ({n}), assigning all to train.")
            train_idx.extend(indices)
            continue
        train_split, temp_split = train_test_split(indices, test_size=0.4, random_state=42, shuffle=True)
        val_split, test_split = train_test_split(temp_split, test_size=0.5, random_state=42, shuffle=True)
        train_idx.extend(train_split)
        val_idx.extend(val_split)
        test_idx.extend(test_split)
        print(f"SNR {snr_val}: {len(train_split)} / {len(val_split)} / {len(test_split)}")
    print(f"Total: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)
    test_dataset  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 实例化 MCDformer（假设其构造函数接受 sig_len 和 num_classes）
    model = MCDformer(sig_len=768, num_classes=12).to(device)

    model.eval()  # 设置为评估模式
    dummy_input = torch.randn(1, 2, 768).to(device)

    # 1. 参数量（单位：M）
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. FLOPs（单位：M）
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        mflops = flops / 1e6   # 转换为百万次操作
    except:
        mflops = 0.0

    # 3. 显存占用（单位：MB）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input)
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        mem_mb = 0

    # 4. Latency 测量（单位：ms）
    if torch.cuda.is_available():
    # 预热
        for _ in range(10):
            _ = model(dummy_input)
        num_runs = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize(device)
        latency_ms = start_event.elapsed_time(end_event) / num_runs
    else:
    # CPU 计时
        for _ in range(10):
            _ = model(dummy_input)
        num_runs = 100
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
        latency_ms = (time.perf_counter() - start) * 1000 / num_runs

    # 5. 计算有效吞吐量（每秒百万次操作，单位：MFLOPs/s，可选）
    if mflops > 0 and latency_ms > 0:
        throughput_mflops = mflops / (latency_ms / 1000.0)  # 单位：MFLOPs/s
    else:
        throughput_mflops = 0.0

    # 输出（所有单位明确标注）
    print(f"  Params (M): {round(total_params, 3)}")
    print(f"  FLOPs (M): {round(mflops, 3)}")          # 总操作数，单位：百万次
    print(f"  GPU Memory (MB): {round(mem_mb, 2)}")
    print(f"  Latency (ms): {round(latency_ms, 3)}")
    print(f"  Throughput (MFLOPs/s): {round(throughput_mflops, 3)}") 

    # # 可选：计算 FLOPs 和参数量（需要安装 thop）
    # try:
    #     from thop import profile, clever_format
    #     input_tensor = torch.randn(1, 2, seq_len).to(device)
    #     flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    #     flops, params = clever_format([flops, params], "%.3f")
    #     print(f"FLOPs: {flops}, Params: {params}")
    # except ImportError:
    #     print("thop not installed, skip FLOPs calculation")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.999))
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    early_stop_counter = 0
    model_path = os.path.join(result_dir, 'best_model_mcdformer.pth')

    print("\n" + "=" * 50)
    print("Starting Modulation Recognition Training with MCDformer...")
    print("=" * 50)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_mod(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch_mod(model, val_loader, criterion, device)
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
            print(f"  ✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 保存训练历史
    with open(os.path.join(result_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_curves(history, result_dir)

    # 测试最佳模型
    model.load_state_dict(torch.load(model_path))
    print("\n========== Final Test Results ==========")
    test_loss, test_acc = eval_epoch_mod(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 详细按 SNR 和调制类型分析
    test_by_snr_and_mod(model, test_loader, device, num_classes=12)

    print(f"\nResults saved to {result_dir}")

def test_MCDFormer_mod():
    """
    测试 MCDFormer 模型的调制识别性能
    """
    # ---------- 配置路径 ----------
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/testdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/mod/MCDFormer/best_model_mcdformer.pth'   # 请修改为实际路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- 加载测试数据集 ----------
    dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=False,
        return_spectrum=False,
        return_angle=False,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        iq_shape='2D'
    )
    print(f"Total test samples: {len(dataset)}")

    # 自动获取 IQ 序列长度
    seq_len = dataset.iq_real.shape[1]
    print(f"Detected IQ sequence length: {seq_len}")

    test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # ---------- 加载模型 ----------
    model = MCDformer(sig_len=768, num_classes=12).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    # 兼容保存时可能包含的额外键
    # if 'model_state_dict' in state_dict:
    #     model.load_state_dict(state_dict['model_state_dict'], strict=True)
    # else:
    #     model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully.")

    # ---------- 定义调制名称 ----------
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    # ---------- 统计字典 ----------
    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)
    mod_correct = defaultdict(lambda: defaultdict(int))
    mod_total = defaultdict(lambda: defaultdict(int))

    # ---------- 逐批次推理 ----------
    with torch.no_grad():
        for iq, label, snr in test_loader:
            iq = iq.to(device)          # (batch, 2, seq_len)
            label = label.to(device)
            output = model(iq)           # (batch, num_classes)
            pred = output.argmax(dim=1)
            for i in range(label.size(0)):
                snr_val = snr[i].item()
                true = label[i].item()
                pred_val = pred[i].item()
                snr_total[snr_val] += 1
                if pred_val == true:
                    snr_correct[snr_val] += 1
                mod_total[true][snr_val] += 1
                if pred_val == true:
                    mod_correct[true][snr_val] += 1

    # ---------- 打印结果 ----------
    print("\n========== 按 SNR 的整体识别准确率 ==========")
    print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
    for snr in sorted(snr_total.keys()):
        acc = snr_correct[snr] / snr_total[snr] * 100
        print(f"{snr:<10} {acc:<15.2f} {snr_total[snr]:<10}")

    print("\n========== 每种调制类型按 SNR 的识别准确率 ==========")
    for mod in range(12):
        print(f"\n调制类型 {mod} ({mod_names[mod]}):")
        print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
        for snr in sorted(mod_total[mod].keys()):
            acc = mod_correct[mod][snr] / mod_total[mod][snr] * 100
            print(f"{snr:<10} {acc:<15.2f} {mod_total[mod][snr]:<10}")

    print("\nTest completed.")

def generate_mcdformer_confusion_matrices(snr_list=[-6, 4]):
    """
    加载 MCDFormer 模型和测试数据，生成指定 SNR 下调制识别的混淆矩阵图片（无标题，字体较大）。

    Args:
        snr_list: list of int/float, 需要生成混淆矩阵的信噪比值，默认 [-2, 4]
    """
    # ---------- 配置路径 ----------
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/dataset/M8newdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/comparemodel/MOD/MCDformer_results/best_model_mcdformer.pth'   # 请修改为实际路径
    result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/confusionmartix/MCD_Former'
    os.makedirs(result_dir, exist_ok=True)

    # ---------- 设备 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- 加载测试数据集（MCDFormer 只需要 IQ、调制标签、SNR）----------
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=False,
        return_spectrum=False,
        return_angle=False,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'          # 返回 (samples, 2, snapshot)
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 获取 IQ 序列长度（快拍数）
    seq_len = test_dataset.iq_real.shape[1]   # 768
    print(f"Detected IQ sequence length: {seq_len}")

    # ---------- 加载模型 ----------
    model = MCDformer(sig_len=seq_len, num_classes=12).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)   # 兼容保存时可能包含额外键
    model.eval()
    print("Model loaded successfully.\n")

    # ---------- 收集指定 SNR 下的预测和真实标签 ----------
    snr_to_labels = {snr: {'true': [], 'pred': []} for snr in snr_list}
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    with torch.no_grad():
        for iq, mod_label, snr in test_loader:
            iq = iq.to(device)
            mod_label = mod_label.to(device)
            snr_batch = snr.cpu().numpy().flatten()

            output = model(iq)
            pred_mod = torch.argmax(output, dim=1).cpu().numpy()
            true_mod = mod_label.cpu().numpy()

            for i in range(len(pred_mod)):
                s = snr_batch[i]
                if s in snr_to_labels:
                    snr_to_labels[s]['true'].append(true_mod[i])
                    snr_to_labels[s]['pred'].append(pred_mod[i])

    # ---------- 绘制并保存混淆矩阵 ----------
    for snr_val, data in snr_to_labels.items():
        true_list = data['true']
        pred_list = data['pred']
        if len(true_list) == 0:
            print(f"Warning: No samples found for SNR={snr_val} dB, skip.")
            continue

        cm = confusion_matrix(true_list, pred_list, labels=range(12))
        # 转换为百分比（按行归一化）
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

        plt.figure(figsize=(12, 10))
        sns.set(font_scale=1.5)   # 字体放大
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=mod_names, yticklabels=mod_names,
                    cbar_kws={'label': 'Accuracy (%)'})
        plt.xlabel('Predicted Modulation')
        plt.ylabel('True Modulation')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(result_dir, f'MCDFormer_confusion_matrix_SNR_{snr_val}dB.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    print("\nAll confusion matrices generated.")


if __name__ == "__main__":
    main_mod()
    # test_MCDFormer_mod()
    # generate_mcdformer_confusion_matrices(snr_list=[-6, 4])