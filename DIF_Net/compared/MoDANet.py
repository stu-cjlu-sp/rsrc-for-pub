import os
import argparse


parser = argparse.ArgumentParser(description="TP_Net")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="2", help='GPU id')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

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
from thop import profile, clever_format
import time

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
from param import * 

#========导入数据包
class H5MultiTaskDataset(Dataset):
    def __init__(self, folder_path, snr_list=None,
                 return_cov=True, return_spectrum=True, return_angle=True,
                 return_iq=True, return_mod=True, return_snr=True,
                 return_fc=False, fc_norm='minmax', iq_shape='2D'):
        """
        数据集类 生成的数据格式：
        - TimeIQ_real/imag: (N, M, snapshot)
        - DOA_labels_music: (N, num_angles)
        - DOA_true: (N,)
        - Mod_labels: (N,)
        - R_features: (N, M, M, 2) 可选
        """
        self.return_cov = return_cov
        self.return_spectrum = return_spectrum
        self.return_angle = return_angle
        self.return_iq = return_iq
        self.return_mod = return_mod
        self.return_snr = return_snr
        self.return_fc = return_fc
        self.fc_norm = fc_norm
        self.iq_shape = iq_shape

        # 数据容器
        self.cov = None
        self.spectrum = None
        self.angle = None
        self.iq_real = None
        self.iq_imag = None
        self.mod = None
        self.snr = None
        self.fc = None

        # 获取 SNR 文件夹
        snr_folders = [f for f in os.listdir(folder_path) if f.startswith('SNR_')]
        if snr_list is not None:
            snr_folders = [f for f in snr_folders 
                           if int(f.split('_')[-1].replace('dB', '')) in snr_list]
        snr_folders.sort()

        cov_list, spectrum_list, angle_list = [], [], []
        iq_real_list, iq_imag_list, mod_list, snr_list_vals = [], [], [], []
        fc_list = []

        for snr_dir in snr_folders:
            snr_value = int(snr_dir.split('_')[-1].replace('dB', ''))
            mat_file = os.path.join(folder_path, snr_dir, 'data.mat')
            with h5py.File(mat_file, 'r') as f:
                # 协方差特征
                if return_cov and 'R_features' in f:
                    R_raw = np.array(f['R_features']).astype(np.float32)
                    if R_raw.ndim == 4:
                        R = np.transpose(R_raw, (3, 0, 1, 2))
                    else:
                        R = R_raw
                    cov_list.append(R)
                elif return_cov:
                    print(f"Warning: {mat_file} does not contain 'R_features'")
                
                # MUSIC 谱
                spec = np.array(f['DOA_labels']).astype(np.float32) #(121,29040)
                if spec.ndim == 2 and spec.shape[0] != (len(cov_list) if cov_list else 0):
                    spec = spec.T#(29040,121)
                spectrum_list.append(spec)
                
                # 真实角度
                angle = np.array(f['DOA_true']).flatten().astype(np.float32)#(29040,)
                angle_list.append(angle)
                
                # 调制类型
                mod = np.array(f['Mod_labels']).flatten().astype(np.int64)#(29040,)
                mod_list.append(mod)
                
                # IQ 数据: (24094, 8, 768)
                iq_real = np.array(f['TimeIQ_real']).astype(np.float32).T   # (N, M, snapshot)
                iq_imag = np.array(f['TimeIQ_imag']).astype(np.float32).T
                if iq_real.ndim == 3:
                    pass
                else:
                    iq_real = iq_real[:, np.newaxis, :]
                    iq_imag = iq_imag[:, np.newaxis, :]
                iq_real_list.append(iq_real)
                iq_imag_list.append(iq_imag)
                
                # SNR 标签
                n = spec.shape[0]
                snr_list_vals.append(np.full(n, snr_value, dtype=np.int32))
                
                # 载频 (可选，新版可能不存在)
                if return_fc and 'fc_labels' in f:
                    fc = np.array(f['fc_labels']).flatten().astype(np.float32)
                    fc_list.append(fc)
                elif return_fc:
                    print(f"Warning: {mat_file} does not contain 'fc_labels', returning zeros.")
                    fc_list.append(np.zeros(n, dtype=np.float32))

        # 合并
        if cov_list:
            self.cov = np.concatenate(cov_list, axis=0)
        if spectrum_list:
            self.spectrum = np.concatenate(spectrum_list, axis=0)
        if angle_list:
            self.angle = np.concatenate(angle_list, axis=0)
        if mod_list:
            self.mod = np.concatenate(mod_list, axis=0)
        if iq_real_list:
            self.iq_real = np.concatenate(iq_real_list, axis=0)
            self.iq_imag = np.concatenate(iq_imag_list, axis=0)
        if snr_list_vals:
            self.snr = np.concatenate(snr_list_vals, axis=0)
        if fc_list:
            self.fc = np.concatenate(fc_list, axis=0)

        self.length = len(self.spectrum) if self.spectrum is not None else 0
        print(f"Total samples loaded: {self.length}")
        if self.return_iq and self.iq_real is not None:
            print(f"IQ shape: {self.iq_real.shape} (samples, M, snapshot)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ret = []
        if self.return_cov and self.cov is not None:
            cov = self.cov[idx]               # (M, M, 2)
            cov = np.transpose(cov, (2, 0, 1)) # (2, M, M)
            ret.append(torch.tensor(cov, dtype=torch.float32))
        if self.return_spectrum and self.spectrum is not None:
            ret.append(torch.tensor(self.spectrum[idx], dtype=torch.float32))
        if self.return_angle and self.angle is not None:
            ret.append(torch.tensor(self.angle[idx], dtype=torch.float32))
        if self.return_iq and self.iq_real is not None:
            real = self.iq_real[idx]   # (M, snapshot)
            imag = self.iq_imag[idx]   # (M, snapshot)
            iq = np.stack([real, imag], axis=0)   # (2, M, snapshot)
            iq = iq.transpose(0, 2, 1)            # (2, snapshot, M)
            if self.iq_shape == '2D':
                ret.append(torch.tensor(iq, dtype=torch.float32))
            else:
                # 复数形式
                iq_complex = real + 1j * imag
                ret.append(torch.tensor(iq_complex, dtype=torch.complex64))
        if self.return_mod and self.mod is not None:
            ret.append(torch.tensor(self.mod[idx], dtype=torch.long))
        if self.return_snr and self.snr is not None:
            ret.append(torch.tensor(self.snr[idx], dtype=torch.long))
        if self.return_fc and self.fc is not None:
            ret.append(torch.tensor(self.fc[idx], dtype=torch.float32))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)


#==========MoDANet的主要模型
class InputBlock(nn.Module):
    def __init__(self, in_ch=2, out_ch=64):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False
        )

        self.pool = nn.MaxPool2d(
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0)
        )

        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.relu(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        # -------------------------
        # Backbone flow
        # -------------------------
        self.backbone = nn.Sequential(

            # 64 x 3 x 1
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # pool 3 x 1, stride (2,1)
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0)
            ),

            # 64 x 3 x 1
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # pool 3 x 1, stride (2,1)
            nn.MaxPool2d(
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0)
            )
        )

        # -------------------------
        # Skip-connection flow
        # -------------------------
        self.skip = nn.Sequential(

            # 64 x 1 x 1
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # pool 5 x 1, stride (4,1)
            nn.MaxPool2d(
                kernel_size=(5, 1),
                stride=(4, 1),
                padding=(2, 0)
            )
        )

    def forward(self, x):
        backbone_out = self.backbone(x)
        skip_out = self.skip(x)
        out = backbone_out + skip_out

        return out

class OutputBlock(nn.Module):
    def __init__(self, in_ch=64, num_classes=12):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=(1, 1),
            bias=False
        )

        self.norm = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(
            in_ch * 6 * 8,
            num_classes
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x

class MoDANet(nn.Module):

    def __init__(
        self,
        num_classes_mod=12,
        num_classes_doa=121
    ):
        super().__init__()

        self.input_block = InputBlock(
            in_ch=2,
            out_ch=64
        )

        self.shared_res = ResBlock(64)

        self.amc_res1 = ResBlock(64)
        self.amc_res2 = ResBlock(64)

        self.doa_res1 = ResBlock(64)
        self.doa_res2 = ResBlock(64)

        self.amc_out = OutputBlock(
            in_ch=64,
            num_classes=num_classes_mod
        )

        self.doa_out = OutputBlock(
            in_ch=64,
            num_classes=num_classes_doa
        )

    def forward(self, x):

        x = self.input_block(x)

        x = self.shared_res(x)

        amc = self.amc_res1(x)
        amc = self.amc_res2(amc)
        amc_out = self.amc_out(amc)

        doa = self.doa_res1(x)
        doa = self.doa_res2(doa)
        doa_out = self.doa_out(doa)

        return amc_out, doa_out

#==========MoDANet的训练参数
def train_epoch(model, loader, optimizer, loss_cls, loss_doa, device,
                doa_grid, lambda_cls=1.0, lambda_doa=1.0):
    """doa_grid: numpy array 或 torch tensor, 形状 (121,) 表示所有离散角度"""
    model.train()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    doa_grid_tensor = torch.tensor(doa_grid, dtype=torch.float, device=device)

    pbar = tqdm(loader, desc="Training", leave=False)
    for angle, iq, mod, snr in pbar:   # spectrum 现为 None 或空，可忽略
        iq = iq.to(device)
        mod = mod.to(device)
        angle = angle.to(device)                 # 真实角度，形状 (B,)

        mod_out, doa_out = model(iq)             # doa_out: (B, 121)

        # ---- 将真实角度转换为类别索引 ----
        # 计算每个角度与网格的绝对差，取最小索引
        target_idx = torch.argmin(torch.abs(angle.unsqueeze(1) - doa_grid_tensor.unsqueeze(0)), dim=1)
        target_idx = target_idx.long()

        loss_c = loss_cls(mod_out, mod)
        loss_d = loss_doa(doa_out, target_idx)   # CrossEntropyLoss
        loss = lambda_cls * loss_c + lambda_doa * loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_cls_loss += loss_c.item() * batch_size
        total_doa_loss += loss_d.item() * batch_size

        pred = torch.argmax(mod_out, dim=1)
        correct = (pred == mod).sum().item()
        total_correct += correct
        pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples
    avg_doa = total_doa_loss / total_samples
    acc = total_correct / total_samples * 100
    return avg_loss, avg_cls, avg_doa, acc

@torch.no_grad()
def eval_epoch(model, loader, loss_cls, loss_doa, device, doa_grid,
               lambda_cls=1.0, lambda_doa=1.0):
    model.eval()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    total_rmse, total_mae = 0.0, 0.0
    doa_grid_tensor = torch.tensor(doa_grid, dtype=torch.float, device=device)

    for angle, iq, mod, snr in loader:
        iq = iq.to(device)
        mod = mod.to(device)
        angle = angle.to(device)

        mod_out, doa_out = model(iq)

        # 角度标签转索引
        target_idx = torch.argmin(torch.abs(angle.unsqueeze(1) - doa_grid_tensor.unsqueeze(0)), dim=1).long()

        loss_c = loss_cls(mod_out, mod)
        loss_d = loss_doa(doa_out, target_idx)
        loss = lambda_cls * loss_c + lambda_doa * loss_d

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_cls_loss += loss_c.item() * batch_size
        total_doa_loss += loss_d.item() * batch_size

        pred_mod = torch.argmax(mod_out, dim=1)
        total_correct += (pred_mod == mod).sum().item()

        # ---- DOA 角度估计（分类 -> 角度） ----
        pred_idx = torch.argmax(doa_out, dim=1)          # (B,)
        pred_angle = doa_grid[pred_idx.cpu().numpy()]    # 映射到角度值
        true_angle_np = angle.cpu().numpy()
        err = pred_angle - true_angle_np
        total_rmse += np.sum(err ** 2)
        total_mae += np.sum(np.abs(err))

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples
    avg_doa = total_doa_loss / total_samples
    acc = total_correct / total_samples * 100
    rmse = np.sqrt(total_rmse / total_samples)
    mae = total_mae / total_samples
    return avg_loss, avg_cls, avg_doa, acc, rmse, mae

@torch.no_grad()
def test_model(model, test_loader, device, doa_grid):
    model.eval()
    total_correct, total_samples = 0, 0
    total_rmse, total_mae = 0.0, 0.0
    for angle, iq, mod, snr in test_loader:
        iq = iq.to(device)
        mod = mod.to(device)
        angle = angle.to(device)

        mod_out, doa_out = model(iq)
        pred_mod = torch.argmax(mod_out, dim=1)
        total_correct += (pred_mod == mod).sum().item()

        pred_idx = torch.argmax(doa_out, dim=1)
        pred_angle = doa_grid[pred_idx.cpu().numpy()]
        true_angle_np = angle.cpu().numpy()
        err = pred_angle - true_angle_np
        total_rmse += np.sum(err ** 2)
        total_mae += np.sum(np.abs(err))
        total_samples += iq.size(0)

    mod_acc = total_correct / total_samples * 100
    rmse = np.sqrt(total_rmse / total_samples)
    mae = total_mae / total_samples
    return mod_acc, rmse, mae

# ==================== 绘图函数（不变） ====================
def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 12))
    plt.subplot(3, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Total Loss'); plt.legend(); plt.title('Total Loss')
    plt.subplot(3, 2, 2)
    plt.plot(epochs, history['train_cls_loss'], label='Train')
    plt.plot(epochs, history['val_cls_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Classification Loss'); plt.legend(); plt.title('Modulation Loss')
    plt.subplot(3, 2, 3)
    plt.plot(epochs, history['train_doa_loss'], label='Train')
    plt.plot(epochs, history['val_doa_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('DOA Loss'); plt.legend(); plt.title('DOA Classification Loss')
    plt.subplot(3, 2, 4)
    plt.plot(epochs, history['train_mod_acc'], label='Train')
    plt.plot(epochs, history['val_mod_acc'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.title('Modulation Accuracy')
    plt.subplot(3, 2, 5)
    plt.plot(epochs, history['val_doa_rmse'], label='RMSE')
    plt.plot(epochs, history['val_doa_mae'], label='MAE')
    plt.xlabel('Epoch'); plt.ylabel('Error (deg)'); plt.legend(); plt.title('DOA Estimation Error')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")

# ==================== 主函数 ====================
def main_MoDANet():
    # ---------- 路径配置 ----------
    # file_path = "/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/CNN_dataset"
    # result_dir = "/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/MoDANet4"
    file_path = "/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/TP_Netdataset"
    result_dir = "/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/MODANE0901/MoDANet2"
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 加载数据集（不返回 MUSIC 谱） ----------
    dataset = H5MultiTaskDataset(
        folder_path=file_path,
        return_cov=False,           # 不需要协方差
        return_spectrum=False,      # 修改：不返回 MUSIC 谱
        return_angle=True,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'
    )
    # ---------- DOA 网格 ----------
    angle_min, angle_max, step = -60, 60, 1
    DOA_grid = np.linspace(angle_min, angle_max, int((angle_max - angle_min)/step) + 1)
    num_doa_classes = len(DOA_grid)

    # ---------- 按 SNR 分层划分 ----------
    snrs = dataset.snr
    snr_to_indices = defaultdict(list)
    for idx, s in enumerate(snrs):
        snr_to_indices[s].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    print("\nSNR-wise sample counts (train/val/test):")
    for s, idxs in snr_to_indices.items():
        n = len(idxs)
        if n < 5:
            print(f"SNR {s} too few samples ({n}), all to train.")
            train_idx.extend(idxs)
            continue
        tr, temp = train_test_split(idxs, test_size=0.4, random_state=42)
        va, te = train_test_split(temp, test_size=0.5, random_state=42)
        train_idx.extend(tr); val_idx.extend(va); test_idx.extend(te)
        print(f"SNR {s}: {len(tr)} / {len(va)} / {len(te)}")
    print(f"Total: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    batch_size = 128
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------- 模型、损失、优化器 ----------
    sample_batch = next(iter(train_loader))   # train_loader 返回 (angle, iq, mod, snr)
    sample_iq = sample_batch[1]               # shape: (batch_size, 2, N, M)
    input_shape = sample_iq.shape[1:]         # (2, N, M)

    # 创建模型
    model = MoDANet(num_classes_mod=12, num_classes_doa=num_doa_classes).to(device)

    # 假设 model 已定义，device 已设置
    dummy_input = torch.randn(1, 2, 768, 8).to(device)

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
    # num_mod_classes = 12   # 修改：数据集调制类型变为 11 类
    # model = TP_Net(num_classes_mod=num_mod_classes, num_classes_doa=num_doa_classes).to(device)
    # dummy_iq = torch.randn(1, 2, 1024, 8).to(device)
    # with torch.no_grad():
    #     _ = model(dummy_iq)   # 触发所有动态创建

    # # 然后计算 FLOPs
    # flops, params = profile(model, inputs=(dummy_iq,), verbose=False)
    # model = TP_Net(num_classes_mod=num_mod_classes, num_classes_doa=num_doa_classes).to(device)

    # # 计算 FLOPs / Params（可选）
    # dummy_iq = torch.randn(1, 2, 1024, 8).to(device)
    # flops, params = profile(model, inputs=(dummy_iq,), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"FLOPs: {flops}, Params: {params}")

    loss_cls = nn.CrossEntropyLoss()
    loss_doa = nn.CrossEntropyLoss()   # 修改：DOA 使用分类损失

    lambda_cls, lambda_doa = 1.0, 1.0
    print(f"Loss weights: cls={lambda_cls}, doa={lambda_doa}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # ---------- 历史记录 ----------
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_doa_loss': [], 'val_doa_loss': [],
        'train_mod_acc': [], 'val_mod_acc': [],
        'val_doa_rmse': [], 'val_doa_mae': []
    }

    # ---------- 训练循环 ----------
    best_val_rmse = float('inf')
    patience, early_stop_cnt = 4, 0
    model_path = os.path.join(result_dir, 'best_model.pth')
    num_epochs = 60

    print("\n" + "="*50)
    print("Starting Joint Training with TP_Net (Corrected DOA Classification)")
    print("="*50)

    for epoch in range(num_epochs):
        train_loss, train_cls, train_doa, train_acc = train_epoch(
            model, train_loader, optimizer, loss_cls, loss_doa, device,
            DOA_grid, lambda_cls, lambda_doa
        )
        val_loss, val_cls, val_doa, val_acc, val_rmse, val_mae = eval_epoch(
            model, val_loader, loss_cls, loss_doa, device, DOA_grid,
            lambda_cls, lambda_doa
        )

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cls_loss'].append(train_cls)
        history['val_cls_loss'].append(val_cls)
        history['train_doa_loss'].append(train_doa)
        history['val_doa_loss'].append(val_doa)
        history['train_mod_acc'].append(train_acc)
        history['val_mod_acc'].append(val_acc)
        history['val_doa_rmse'].append(val_rmse)
        history['val_doa_mae'].append(val_mae)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.6f} (Cls: {train_cls:.6f}) Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.6f} (Cls: {val_cls:.6f}) Acc: {val_acc:.2f}%")
        print(f"  DOA - Train Loss: {train_doa:.6f}, Val Loss: {val_doa:.6f}")
        print(f"        Val DOA RMSE: {val_rmse:.4f}°, MAE: {val_mae:.4f}°")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), model_path)
            early_stop_cnt = 0
            print(f"  ✓ Saved best model (Val DOA RMSE: {best_val_rmse:.4f}°)")
        else:
            early_stop_cnt += 1
        if early_stop_cnt >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ---------- 保存历史与绘图 ----------
    with open(os.path.join(result_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_training_history(history, result_dir)

    # ---------- 测试最佳模型 ----------
    model.load_state_dict(torch.load(model_path))
    test_acc, test_rmse, test_mae = test_model(model, test_loader, device, DOA_grid)

    print("\n========== Final Test Results ==========")
    print(f"Test Modulation Accuracy: {test_acc:.2f}%")
    print(f"Test DOA RMSE: {test_rmse:.4f}°, MAE: {test_mae:.4f}°")
    print(f"\nAll results saved to {result_dir}")

# ================= 辅助函数：计算验证集指标 =================
def compute_metrics_tpnet(model, loader, device, DOA_grid):
    """计算验证集的 RMSE、MAE 和调制准确率（用于早期验证）"""
    model.eval()
    all_pred_angles, all_true_angles = [], []
    all_pred_mods, all_true_mods = [], []
    with torch.no_grad():
        for batch in loader:
            true_angle = batch[0].to(device)
            iq = batch[1].to(device)
            mod_label = batch[2].to(device)
            mod_out, doa_out = model(iq)

            pred_mod = torch.argmax(mod_out, dim=1).cpu().numpy()
            true_mod = mod_label.cpu().numpy()
            pred_idx = torch.argmax(doa_out, dim=1).cpu().numpy()
            pred_angle = DOA_grid[pred_idx]
            true_angle_np = true_angle.cpu().numpy()

            all_pred_angles.extend(pred_angle)
            all_true_angles.extend(true_angle_np)
            all_pred_mods.extend(pred_mod)
            all_true_mods.extend(true_mod)

    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    err = all_pred_angles - all_true_angles
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    mod_acc = np.mean(np.array(all_pred_mods) == np.array(all_true_mods)) * 100
    return rmse, mae, mod_acc

def test_MoDANet(
    # test_data_path='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/CNNtestdata',
    # model_path='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/MoDANet4/best_model.pth',
    test_data_path='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/TPNettestdata2',
    model_path='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/MODANE0901/MoDANet2/best_model.pth',
    # result_dir='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/MoDANet4',
    batch_size=64,
    num_workers=4,
    device=None
):
    """
    测试 TP-Net 在测试集上的调制识别准确率和 DOA 估计 RMSE/MAE。
    DOA 估计使用 argmax（离散分类），稳定且符合训练目标。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ================= DOA 网格 =================
    DOA_grid = np.arange(-60, 61, 1)          # shape (121,)
    num_doa_classes = len(DOA_grid)

    # ================= 数据集 =================
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=False,
        return_spectrum=False,
        return_angle=True,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ================= 模型加载 =================
    model = MoDANet(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully")

    # ================= 统计容器 =================
    snr_total_mod = defaultdict(int)
    snr_correct_mod = defaultdict(int)

    mod_total = {i: defaultdict(int) for i in range(12)}
    mod_correct = {i: defaultdict(int) for i in range(12)}

    all_pred_angles = []
    all_true_angles = []
    all_snrs = []

    # ================= 推理循环 =================
    with torch.no_grad():
        for batch in test_loader:
            # 返回顺序：true_angle, iq, mod_label, snr
            true_angle = batch[0].to(device)
            iq = batch[1].to(device)
            mod_label = batch[2].to(device)
            snr_batch = batch[3].cpu().numpy()

            # 前向传播
            mod_out, doa_out = model(iq)

            # ---------- 调制识别统计 ----------
            pred_mod = torch.argmax(mod_out, dim=1).cpu().numpy()
            true_mod = mod_label.cpu().numpy()

            for i in range(len(pred_mod)):
                s = int(round(snr_batch[i]))
                snr_total_mod[s] += 1
                if pred_mod[i] == true_mod[i]:
                    snr_correct_mod[s] += 1
                mod_total[true_mod[i]][s] += 1
                if pred_mod[i] == true_mod[i]:
                    mod_correct[true_mod[i]][s] += 1

            # ---------- DOA 估计（使用 argmax 代替 soft-argmax） ----------
            pred_idx = torch.argmax(doa_out, dim=1)          # (B,)
            pred_angle_np = DOA_grid[pred_idx.cpu().numpy()]  # 映射到角度
            true_angle_np = true_angle.cpu().numpy()

            for i in range(len(pred_angle_np)):
                all_pred_angles.append(pred_angle_np[i])
                all_true_angles.append(true_angle_np[i])
                all_snrs.append(int(round(snr_batch[i])))

    # ================= 转换为 numpy =================
    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)

    # ================= 整体性能 =================
    mod_acc = (
        sum(snr_correct_mod.values())
        / sum(snr_total_mod.values())
        * 100
    )

    doa_error = all_pred_angles - all_true_angles
    doa_rmse = np.sqrt(np.mean(doa_error ** 2))
    doa_mae = np.mean(np.abs(doa_error))

    print("\n========== Overall Performance ==========")
    print(f"Modulation Accuracy: {mod_acc:.2f}%")
    print(f"DOA RMSE: {doa_rmse:.4f} deg")
    print(f"DOA MAE : {doa_mae:.4f} deg")

    # ================= SNR 维度统计 =================
    print("\n========== SNR-wise Performance ==========")
    print(f"{'SNR':<8}{'MOD Acc':<12}{'RMSE':<12}{'MAE':<12}{'N':<8}")
    unique_snrs = sorted(np.unique(all_snrs))
    for s in unique_snrs:
        mask = (all_snrs == s)
        err = all_pred_angles[mask] - all_true_angles[mask]
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(np.abs(err))
        acc = snr_correct_mod[s] / snr_total_mod[s] * 100
        print(f"{s:<8}{acc:<12.2f}{rmse:<12.4f}{mae:<12.4f}{np.sum(mask):<8}")

    # ================= 保存结果 =================
    os.makedirs(result_dir, exist_ok=True)

    results = {
        "overall": {
            "mod_acc": mod_acc,
            "doa_rmse": doa_rmse,
            "doa_mae": doa_mae
        },
        "pred_angle": all_pred_angles,
        "true_angle": all_true_angles,
        "snr": all_snrs
    }

    with open(os.path.join(result_dir, "test_results_tpnet.pkl"), "wb") as f:
        pickle.dump(results, f)

    # ================= 绘制误差直方图 =================
    plt.figure(figsize=(8, 5))
    plt.hist(doa_error, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("DOA Error (deg)")
    plt.ylabel("Count")
    plt.title("TP-Net DOA Error Distribution (argmax)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(result_dir, "tpnet_error_hist.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {result_dir}")
    return results

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    set_seed(46)
    # main_MoDANet()
    test_MoDANet()
    # save_tpnet_scatter_by_snr(target_snr=0, save_filename='scatter_0dB_TP_Net.png') #散点图
    # generate_tpnet_confusion_matrices(snr_list=[-6, 4])




