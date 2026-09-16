import os
import argparse

parser = argparse.ArgumentParser(description="CNN")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
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
from collections import defaultdict

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
from tool import * 
from thop import profile, clever_format

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.in_c = in_channel
        self.out_c = out_channel

        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c)
        )

    def forward(self, x):
        """
        x: [batchsize, C, H, W]
        """
        x = self.conv_block(x)

        return x


class MultiScaleModule(nn.Module):
    def __init__(self, out_channel):
        super(MultiScaleModule, self).__init__()
        self.out_c = out_channel

        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )

    def forward(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        x = torch.cat([y1, y2, y3], dim=1)

        return x


class TinyMLP(nn.Module):
    def __init__(self, N):
        super(TinyMLP, self).__init__()
        self.N = N

        self.mlp = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class AdaCorrModule(nn.Module):
    def __init__(self, N):
        super(AdaCorrModule, self).__init__()
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

    def forward(self, x):
        # x:[N, C_out, 1, W]
        x_init = copy.deepcopy(x)
        x = torch.fft.fft(x, dim=-1)
        X_re = torch.real(x)
        X_im = torch.imag(x)
        h_re = self.Re(X_re)
        h_im = self.Im(X_im)
        # x:[N, C_out, 1, W]_complex
        x = torch.mul(h_re, X_re) + 1j * torch.mul(h_im, X_im)
        x = torch.real(torch.fft.ifft(x, dim=-1))
#         x = x / x.norm(p=2, dim=-1, keepdim=True)
#         x_init = x_init / x_init.norm(p=2, dim=-1, keepdim=True)
        x = x + x_init
        
        return x


class FeaFusionModule(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(FeaFusionModule, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value_heads)
        shape = context.size()
        context = context.contiguous().view(shape[0], -1, shape[-1])
        return context


class AMC_Net(nn.Module):
    def __init__(self,
                 num_classes=26,
                 sig_len=1024,
                 extend_channel=36,
                 latent_dim=512,
                 num_heads=2,
                 conv_chan_list=None):
        super(AMC_Net, self).__init__()
        self.sig_len = sig_len
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = conv_chan_list

        if self.conv_chan_list is None:
            self.conv_chan_list = [36, 64, 128, 256]
        self.stem_layers_num = len(self.conv_chan_list) - 1

        self.ACM = AdaCorrModule(self.sig_len)
        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM = FeaFusionModule(self.num_heads, self.sig_len, self.sig_len)

        self.Conv_stem = nn.Sequential()

        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(f'conv_stem_{t}',
                                      Conv_Block(
                                          self.conv_chan_list[t],
                                          self.conv_chan_list[t + 1])
                                      )

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = x.unsqueeze(1)
        x = self.ACM(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = self.FFM(x.squeeze(2))
        x = self.GAP(x)
        y = self.classifier(x.squeeze(2))
        return y


def train_epoch_mod(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for iq, label, snr in loader:
        iq = iq.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(iq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    return total_loss / len(loader), correct / total * 100


def eval_epoch_mod(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for iq, label, snr in loader:
            iq = iq.to(device)
            label = label.to(device)
            output = model(iq)
            loss = criterion(output, label)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return total_loss / len(loader), correct / total * 100


def plot_curves(history, save_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def test_by_snr_and_mod(model, loader, device, num_classes=12):
    model.eval()
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]
    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)
    mod_correct = defaultdict(lambda: defaultdict(int))
    mod_total = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for iq, label, snr in loader:
            iq = iq.to(device)
            label = label.to(device)
            output = model(iq)
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

    print("\n========== 按SNR的识别准确率 ==========")
    print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
    for snr in sorted(snr_total.keys()):
        acc = snr_correct[snr] / snr_total[snr] * 100
        print(f"{snr:<10} {acc:<15.2f} {snr_total[snr]:<10}")

    print("\n========== 每种调制类型按SNR的识别准确率 ==========")
    for mod in range(num_classes):
        print(f"\n调制类型 {mod} ({mod_names[mod]}):")
        print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
        for snr in sorted(mod_total[mod].keys()):
            acc = mod_correct[mod][snr] / mod_total[mod][snr] * 100
            print(f"{snr:<10} {acc:<15.2f} {mod_total[mod][snr]:<10}")


def main_mod():
    # ---------- 路径配置 ----------
    result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/mod/AMC-Net_0901'         
    os.makedirs(result_dir, exist_ok=True)
    file_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/dataset'     

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 256
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    patience = 5

    # 加载数据集（只取IQ、调制标签、SNR）
    dataset = H5MultiTaskDataset(
        folder_path=file_path,
        return_cov=False,
        return_spectrum=False,
        return_angle=False,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'
    )
    # ---------------- 按SNR分组划分训练/验证/测试集 ----------------
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

    # ---------------- 模型（SEFEFeatureExtractor） ----------------
    # model = AMC_Net(num_classes=12).to(device)
    model = AMC_Net(
        num_classes=12,           # 根据你的实际调制类别数修改
        sig_len=768,              # 必须与输入信号长度一致
        extend_channel=36,
        latent_dim=512,
        num_heads=2,
        conv_chan_list=[36, 64, 128, 256]
    ).to(device)
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
    # 计算FLOPs和参数量（可选）
    # input_tensor = torch.randn(1, 2, 768).to(device)
    # flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"FLOPs: {flops}, Params: {params}")

    # ---------------- 损失、优化器、调度器 ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    early_stop_counter = 0
    model_path = os.path.join(result_dir, 'best_model_mod.pth')

    print("\n" + "=" * 50)
    print("Starting Modulation Recognition Training with AMC-net...")
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

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(model_path))
    print("\n========== Final Test Results ==========")
    test_loss, test_acc = eval_epoch_mod(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 详细按SNR和调制类型分析
    test_by_snr_and_mod(model, test_loader, device, num_classes=12)

    print(f"\nResults saved to {result_dir}")

def test_AMCNet():
    """
    测试 AMC-Net (OverallNetwork) 模型的调制识别性能
    """
    # ---------- 配置路径 ----------
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/testdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/mod/AMC-Net/best_model_mod.pth'  # 请修改为实际路径
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
        iq_shape='2D'          # 返回 (samples, 2, snapshot) 或 (samples, 2, snapshot, 1)
    )
    print(f"Total test samples: {len(dataset)}")

    # 自动获取 IQ 序列长度
    # 注意：dataset.iq_real 可能是 (samples, snapshot) 或 (samples, 2, snapshot)
    iq_shape = dataset[0][0].shape
    if len(iq_shape) == 3:
        seq_len = iq_shape[2]   # (2, T, 1) 或 (2, T)
    else:
        seq_len = iq_shape[1]   # (2, T)
    print(f"Detected IQ sequence length: {seq_len}")

    test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # ---------- 加载模型 ----------
    model = OverallNetwork(num_classes=12).to(device)
    state_dict = torch.load(model_path, map_location=device)
    # 注意：训练时保存的是整个 model.state_dict()，直接加载
    model.load_state_dict(state_dict, strict=False)
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
            # 数据形状兼容：若为 (batch,2,T,1) 则压缩为 (batch,2,T)
            if iq.dim() == 4:
                iq = iq.squeeze(-1)
            iq = iq.to(device)
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

def generate_amcnet_confusion_matrices(snr_list=[-6, 4]):
    """
    加载 AMC-Net 模型和测试数据，生成指定 SNR 下调制识别的混淆矩阵图片（无标题，字体较大）。

    Args:
        snr_list: list of int/float, 需要生成混淆矩阵的信噪比值，默认 [-2, 4]
    """
    # ---------- 配置路径 ----------
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/dataset/M8newdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/mod/AMC-Net/best_model_mod.pth' 
    # result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/confusionmartix/AMC-Net'
    # os.makedirs(result_dir, exist_ok=True)

    # ---------- 设备 ----------
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
        iq_shape='2D'          # 返回 (samples, 2, snapshot) 或 (samples, 2, snapshot, 1)
    )
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # 自动获取 IQ 序列长度（仅用于信息输出）
    iq_shape = dataset[0][0].shape
    if len(iq_shape) == 3:
        seq_len = iq_shape[2]
    else:
        seq_len = iq_shape[1]
    print(f"Detected IQ sequence length: {seq_len}")

    # ---------- 加载模型 ----------
    model = OverallNetwork(num_classes=12).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)   # 兼容保存格式
    model.eval()
    print("Model loaded successfully.\n")

    # ---------- 收集指定 SNR 下的预测和真实标签 ----------
    snr_to_labels = {snr: {'true': [], 'pred': []} for snr in snr_list}
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    with torch.no_grad():
        for iq, label, snr in test_loader:
            # 数据形状兼容：若为 (batch,2,T,1) 则压缩为 (batch,2,T)
            if iq.dim() == 4:
                iq = iq.squeeze(-1)
            iq = iq.to(device)
            label = label.to(device)
            snr_batch = snr.cpu().numpy().flatten()

            output = model(iq)
            pred = output.argmax(dim=1).cpu().numpy()
            true = label.cpu().numpy()

            for i in range(len(pred)):
                s = snr_batch[i]
                if s in snr_to_labels:
                    snr_to_labels[s]['true'].append(true[i])
                    snr_to_labels[s]['pred'].append(pred[i])

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

        save_path = os.path.join(result_dir, f'AMCNet_confusion_matrix_SNR_{snr_val}dB.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    print("\nAll confusion matrices generated.")

if __name__ == "__main__":
    # 训练模型
    main_mod()
    # test_AMCNet()
    # generate_amcnet_confusion_matrices(snr_list=[-6, 4])
