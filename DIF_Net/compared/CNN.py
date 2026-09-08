import os
import argparse

parser = argparse.ArgumentParser(description="CNN")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="3", help='GPU id')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 再导入 PyTorch
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

# from param import *

class H5MultiTaskDataset(Dataset):
    """
    读取由 Matlab 保存的 data.mat 文件（格式：-v7.3)
    支持返回协方差、MUSIC谱、角度、IQ、调制类型、SNR、载频等。
    
    Matlab 保存变量约定（必须一致）：
        R_features    : (N, M, M, 2)  实部+虚部
        DOA_labels    : (N, num_angles)  MUSIC谱
        DOA_true      : (N,)           真实角度
        Mod_labels    : (N,)           调制类型标签 (0~11)
        TimeIQ_real   : (N, snapshot)  第一阵元实部
        TimeIQ_imag   : (N, snapshot)  第一阵元虚部
        fc_labels     : (N,)           载频 (Hz)
    """
    def __init__(self, folder_path, snr_list=None,
                 return_cov=True, return_spectrum=True, return_angle=True,
                 return_iq=True, return_mod=True, return_snr=True,
                 return_fc=False, fc_norm='minmax', iq_shape='2D'):
        """
        Args:
            folder_path (str): 包含 SNR_* 子文件夹的根目录
            snr_list (list): 要加载的 SNR 值列表，如 [-2, 0, 2]，None 表示全部
            return_* (bool): 控制返回哪些数据
            fc_norm (str): 载频归一化方式 'minmax' | 'logminmax' | 'zscore' | None
            iq_shape (str): '2D' 返回 (2, snapshot, 1)，'complex' 返回复数张量
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

        # 存储数据
        self.cov = None
        self.spectrum = None
        self.angle = None
        self.iq_real = None
        self.iq_imag = None
        self.mod = None
        self.snr = None
        self.fc = None          # 归一化后的值（或原始值）
        
        # 归一化参数（用于反归一化）
        self.fc_min = None
        self.fc_max = None
        self.fc_mean = None
        self.fc_std = None

        # 获取所有 SNR 子文件夹
        snr_folders = [f for f in os.listdir(folder_path) if f.startswith('SNR_')]
        if snr_list is not None:
            snr_folders = [f for f in snr_folders 
                           if int(f.split('_')[-1].replace('dB', '')) in snr_list]
        snr_folders.sort()

        cov_list, spectrum_list, angle_list = [], [], []
        iq_real_list, iq_imag_list, mod_list, snr_list_vals, fc_list = [], [], [], [], []

        for snr_dir in snr_folders:
            snr_value = int(snr_dir.split('_')[-1].replace('dB', ''))
            mat_file = os.path.join(folder_path, snr_dir, 'data.mat')
            with h5py.File(mat_file, 'r') as f:
                # ---------- 读取各变量（维度与 Matlab 保存一致） ----------
                R = np.array(f['R_features']).astype(np.float32)          # (N, M, M, 2)
                R = np.transpose(R,(3,1,2,0))
                spec = np.array(f['DOA_labels']).astype(np.float32)       # (N, num_angles)
                spec = spec.T
                angle = np.array(f['DOA_true']).flatten().astype(np.float32)
                mod = np.array(f['Mod_labels']).flatten().astype(np.int64)
                iq_real = np.array(f['TimeIQ_real']).astype(np.float32)   # (N, snapshot)
                iq_imag = np.array(f['TimeIQ_imag']).astype(np.float32)   # (N, snapshot)
                iq_real = np.transpose(iq_real,(2,1,0))
                iq_imag = np.transpose(iq_imag,(2,1,0))
                # 载频（兼容旧版本无 fc_labels）
                if 'fc_labels' in f:
                    fc = np.array(f['fc_labels']).flatten().astype(np.float32)
                else:
                    fc = np.zeros(R.shape[0], dtype=np.float32)
                    print(f"Warning: {snr_dir} does not contain 'fc_labels'.")

                n = R.shape[0]
                # 一致性检查
                assert spec.shape[0] == n, "DOA_labels sample count mismatch"
                assert angle.shape[0] == n, "DOA_true sample count mismatch"
                assert mod.shape[0] == n, "Mod_labels sample count mismatch"
                assert iq_real.shape[0] == n, "TimeIQ_real sample count mismatch"
                assert fc.shape[0] == n, "fc_labels sample count mismatch"

                cov_list.append(R)
                spectrum_list.append(spec)
                angle_list.append(angle)
                mod_list.append(mod)
                iq_real_list.append(iq_real)
                iq_imag_list.append(iq_imag)
                fc_list.append(fc)
                snr_list_vals.append(np.full(n, snr_value, dtype=np.int32))

        # 合并所有数据
        if cov_list:
            self.cov = np.concatenate(cov_list, axis=0)
            self.spectrum = np.concatenate(spectrum_list, axis=0)
            self.angle = np.concatenate(angle_list, axis=0)
            self.mod = np.concatenate(mod_list, axis=0)
            self.iq_real = np.concatenate(iq_real_list, axis=0)
            self.iq_imag = np.concatenate(iq_imag_list, axis=0)
            self.snr = np.concatenate(snr_list_vals, axis=0)
            self.fc = np.concatenate(fc_list, axis=0)   # 原始载频（Hz）
        else:
            raise ValueError(f"No data found in {folder_path}")

        # ---------- 载频归一化 ----------
        if self.return_fc and self.fc_norm is not None:
            if self.fc_norm == 'minmax':
                self.fc_min = self.fc.min()
                self.fc_max = self.fc.max()
                if self.fc_max - self.fc_min < 1e-12:
                    self.fc_min = 0.0
                    self.fc_max = 1.0
                self.fc = (self.fc - self.fc_min) / (self.fc_max - self.fc_min)
                print(f"MinMax norm: fc_min={self.fc_min:.2e}, fc_max={self.fc_max:.2e}")
            elif self.fc_norm == 'logminmax':
                fc_log = np.log10(self.fc + 1e-12)
                self.fc_min = fc_log.min()
                self.fc_max = fc_log.max()
                self.fc = (fc_log - self.fc_min) / (self.fc_max - self.fc_min)
                print(f"LogMinMax norm: log10(fc) min={self.fc_min:.4f}, max={self.fc_max:.4f}")
            elif self.fc_norm == 'zscore':
                self.fc_mean = self.fc.mean()
                self.fc_std = self.fc.std()
                self.fc = (self.fc - self.fc_mean) / self.fc_std
                print(f"Z-score norm: fc_mean={self.fc_mean:.2e}, fc_std={self.fc_std:.2e}")
            else:
                raise ValueError(f"Unknown fc_norm: {self.fc_norm}")
        elif self.return_fc:
            print("FC labels are used in original scale (no normalization).")

        self.length = len(self.cov)
        print(f"Total samples loaded: {self.length}")
        if self.return_iq:
            print(f"IQ shape: {self.iq_real.shape} (samples, snapshot)")

    # ---------- 反归一化工具（NumPy 版本） ----------
    def get_fc_denorm(self, fc_normed):
        """将归一化后的载频转换回原始值（用于评估 RMSE 等）"""
        if self.fc_norm == 'minmax':
            return fc_normed * (self.fc_max - self.fc_min) + self.fc_min
        elif self.fc_norm == 'logminmax':
            log_fc = fc_normed * (self.fc_max - self.fc_min) + self.fc_min
            return 10 ** log_fc
        elif self.fc_norm == 'zscore':
            return fc_normed * self.fc_std + self.fc_mean
        else:
            return fc_normed   # 未归一化

    # ---------- 反归一化工具（PyTorch 张量版本，保留计算图） ----------
    def denorm_tensor(self, fc_normed_tensor):
        if self.fc_norm == 'minmax':
            return fc_normed_tensor * (self.fc_max - self.fc_min) + self.fc_min
        elif self.fc_norm == 'logminmax':
            log_fc = fc_normed_tensor * (self.fc_max - self.fc_min) + self.fc_min
            return 10 ** log_fc
        elif self.fc_norm == 'zscore':
            return fc_normed_tensor * self.fc_std + self.fc_mean
        else:
            return fc_normed_tensor

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ret = []
        if self.return_cov:
            cov = self.cov[idx]                     # (M, M, 2)
            cov = np.transpose(cov, (2, 0, 1))      # (2, M, M)  通道在前
            ret.append(torch.tensor(cov, dtype=torch.float32))
        if self.return_spectrum:
            ret.append(torch.tensor(self.spectrum[idx], dtype=torch.float32))
        if self.return_angle:
            ret.append(torch.tensor(self.angle[idx], dtype=torch.float32))
        if self.return_iq:
            if self.iq_shape == '2D':
                iq = np.stack([self.iq_real[idx], self.iq_imag[idx]], axis=0)  # (2, snapshot)
                iq = iq[..., np.newaxis]            # (2, snapshot, 1)
                ret.append(torch.tensor(iq, dtype=torch.float32))
            else:   # 'complex'
                iq_complex = self.iq_real[idx] + 1j * self.iq_imag[idx]
                iq_complex = iq_complex[..., np.newaxis] if iq_complex.ndim == 1 else iq_complex
                ret.append(torch.tensor(iq_complex, dtype=torch.complex64))
        if self.return_mod:
            ret.append(torch.tensor(self.mod[idx], dtype=torch.long))
        if self.return_snr:
            ret.append(torch.tensor(self.snr[idx], dtype=torch.long))
        if self.return_fc:
            ret.append(torch.tensor(self.fc[idx], dtype=torch.float32))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

class DOA_CNN(nn.Module):
    """Deep Networks for DOA Estimation (分类版本)"""
    def __init__(self, num_classes=81, input_channels=3, input_size=8):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)

        # 动态计算全连接输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self._conv_forward(dummy)
            fc_in = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(fc_in, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

    def _conv_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        logits = self.fc4(x)          # 不经过 Sigmoid，留给 CrossEntropyLoss
        return logits

@torch.no_grad()
def valid_epoch_doa(model, loader, criterion, device, doa_grid):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_rmse = 0.0
    total_mae = 0.0

    for cov, spectrum, angle, _ in loader:   # 根据 __getitem__ 顺序：cov, spectrum, angle, snr
        cov = cov.to(device)                 # (B, C, M, M)  C=3
        spectrum = spectrum.to(device)       # (B, num_classes)  one-hot float
        angle = angle.to(device)             # (B,)  真实角度（浮点数）

        outputs = model(cov)                 # (B, num_classes) logits
        loss = criterion(outputs, spectrum)  # BCEWithLogitsLoss

        batch_size = cov.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # 预测：取概率最大的格子
        probs = torch.sigmoid(outputs)
        pred_idx = torch.argmax(probs, dim=1)          # (B,)
        pred_angle = doa_grid[pred_idx.cpu().numpy()]  # 映射到角度值
        true_angle_np = angle.cpu().numpy()
        err = pred_angle - true_angle_np
        total_rmse += np.sum(err ** 2)
        total_mae += np.sum(np.abs(err))

    avg_loss = total_loss / total_samples
    rmse = np.sqrt(total_rmse / total_samples)
    mae = total_mae / total_samples
    return avg_loss, rmse, mae

@torch.no_grad()
def test_epoch_doa(model, loader, criterion, device, doa_grid):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_rmse = 0.0
    total_mae = 0.0

    for cov, spectrum, angle, _ in loader:
        cov = cov.to(device)
        spectrum = spectrum.to(device)
        angle = angle.to(device)

        outputs = model(cov)
        loss = criterion(outputs, spectrum)

        batch_size = cov.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(outputs)
        pred_idx = torch.argmax(probs, dim=1)
        pred_angle = doa_grid[pred_idx.cpu().numpy()]
        true_angle_np = angle.cpu().numpy()
        err = pred_angle - true_angle_np
        total_rmse += np.sum(err ** 2)
        total_mae += np.sum(np.abs(err))

    avg_loss = total_loss / total_samples
    rmse = np.sqrt(total_rmse / total_samples)
    mae = total_mae / total_samples
    return avg_loss, rmse, mae

def train_epoch_doa(model, loader, optimizer, criterion, device, doa_grid):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_rmse = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for cov, spectrum, angle, _ in pbar:
        cov = cov.to(device)
        spectrum = spectrum.to(device)
        angle = angle.to(device)

        outputs = model(cov)
        loss = criterion(outputs, spectrum)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = cov.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(outputs)
        pred_idx = torch.argmax(probs, dim=1)
        pred_angle = doa_grid[pred_idx.cpu().numpy()]
        true_angle_np = angle.cpu().numpy()
        err = pred_angle - true_angle_np
        total_rmse += np.sum(err ** 2)

        pbar.set_postfix(loss=loss.item(), rmse=np.sqrt(np.mean(err**2)))

    avg_loss = total_loss / total_samples
    rmse = np.sqrt(total_rmse / total_samples)
    return avg_loss, rmse


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def main_doa():
    # ---------- 路径配置 ----------
    result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/DOA/CNN'
    file_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/CNN_dataset'
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- 超参数 ----------
    batch_size = 32
    num_epochs = 200
    lr = 1e-3
    weight_decay = 1e-4
    patience = 10

    angle_min, angle_max, step = -60, 60, 1
    num_classes = int((angle_max - angle_min) / step) + 1
    DOA_grid = np.linspace(angle_min, angle_max, num_classes)

    # ---------- 加载数据集 ----------
    dataset = H5MultiTaskDataset(
        folder_path=file_path,
        return_cov=True,
        return_spectrum=True,
        return_angle=True,
        return_iq=False,
        return_mod=False,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'
    )
    print(f"Total samples: {len(dataset)}")

    # 按 SNR 分层划分
    snrs = dataset.snr
    snr_to_indices = defaultdict(list)
    for idx, snr_val in enumerate(snrs):
        snr_to_indices[snr_val].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for snr_val, indices in snr_to_indices.items():
        n = len(indices)
        if n < 5:
            train_idx.extend(indices)
            continue
        train_split, temp = train_test_split(indices, test_size=0.4, random_state=42)
        val_split, test_split = train_test_split(temp, test_size=0.5, random_state=42)
        train_idx.extend(train_split)
        val_idx.extend(val_split)
        test_idx.extend(test_split)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------- 模型 ----------
    model = DOA_CNN(num_classes=num_classes, input_channels=3, input_size=8).to(device)

    model.eval()  # 设置为评估模式
    dummy_input = torch.randn(1, 3, 8, 8).to(device)

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


    # # 计算 FLOPs 和参数量（可选）
    # dummy = torch.randn(1, 3, 8, 8).to(device)
    # flops, params = profile(model, inputs=(dummy,), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"FLOPs: {flops}, Params: {params}")

    # ---------- 损失、优化器 ----------
    criterion = nn.BCEWithLogitsLoss()   # 论文指定
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ---------- 历史记录 ----------
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'val_mae': [], 'lr': []
    }

    best_val_rmse = float('inf')
    early_stop_cnt = 0
    model_path = os.path.join(result_dir, 'best_model_doa.pth')

    print("\n" + "="*50)
    print("Starting DOA Estimation Training (BCEWithLogitsLoss)")
    print("="*50)

    for epoch in range(num_epochs):
        train_loss, train_rmse = train_epoch_doa(
            model, train_loader, optimizer, criterion, device, DOA_grid
        )
        val_loss, val_rmse, val_mae = valid_epoch_doa(
            model, val_loader, criterion, device, DOA_grid
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.6f}, RMSE: {train_rmse:.4f}°")
        print(f"  Val   - Loss: {val_loss:.6f}, RMSE: {val_rmse:.4f}°, MAE: {val_mae:.4f}°")
        print(f"  LR: {current_lr:.2e}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': best_val_rmse,
                'val_loss': val_loss
            }, model_path)
            early_stop_cnt = 0
            print(f"  ✓ Saved best model (Val RMSE: {best_val_rmse:.4f}°)")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 保存训练历史
    with open(os.path.join(result_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_training_curves_doa(history, result_dir)   # 需自行定义绘图函数

    # 加载最佳模型测试
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Best Val RMSE: {checkpoint['val_rmse']:.4f}°")

    test_loss, test_rmse, test_mae = test_epoch_doa(
        model, test_loader, criterion, device, DOA_grid
    )
    print("\n========== Final Test Results ==========")
    print(f"Test Loss: {test_loss:.6f}, Test RMSE: {test_rmse:.4f}°, Test MAE: {test_mae:.4f}°")

    print(f"\nResults saved to {result_dir}")


def test_model_doa():
    # ================= 路径 =================
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/CNNtestdata'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/DOA/CNN/best_model_doa.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ================= DOA grid =================
    angle_min, angle_max, step = -60, 60, 1
    num_classes = int((angle_max - angle_min) / step) + 1
    DOA_grid = np.linspace(angle_min, angle_max, num_classes)

    # ================= 数据集 =================
    print("Loading test dataset...")
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True,
        return_spectrum=False,      # 不需要谱标签
        return_angle=True,
        return_iq=False,
        return_mod=False,
        return_snr=True,
        iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # ================= 模型（输入通道改为3） =================
    model = DOA_CNN(num_classes=num_classes, input_channels=3, input_size=8).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)   # 不用 strict=False，保持严格匹配
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.\n")

    # ================= 统计变量 =================
    all_pred_angles = []
    all_true_angles = []
    all_snrs = []

    # ================= 推理 =================
    with torch.no_grad():
        for batch in test_loader:
            # dataset 返回顺序：(cov, angle, snr)  因为 spectrum=False
            cov, true_angle, snr = batch

            cov = cov.to(device).float()

            # ===== forward =====
            output = model(cov)                     # (B, num_classes) logits

            # BCE 训练：使用 sigmoid 得到概率
            prob = torch.sigmoid(output)            # (B, num_classes) 各格子的独立概率
            pred_class = torch.argmax(prob, dim=1)  # 取最大概率的格子

            pred_angle = DOA_grid[pred_class.cpu().numpy()]
            true_angle = true_angle.numpy()
            snr = snr.numpy()

            all_pred_angles.extend(pred_angle)
            all_true_angles.extend(true_angle)
            all_snrs.extend(snr)

    # ================= 整体性能 =================
    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)

    overall_rmse = np.sqrt(np.mean((all_pred_angles - all_true_angles) ** 2))
    overall_mae = np.mean(np.abs(all_pred_angles - all_true_angles))

    print("\n========== Overall Test Performance ==========")
    print(f"RMSE: {overall_rmse:.4f}°")
    print(f"MAE : {overall_mae:.4f}°\n")

    # ================= 分 SNR =================
    unique_snrs = np.unique(all_snrs)
    print("========== SNR-wise Performance ==========")
    print(f"{'SNR (dB)':<10} {'RMSE (deg)':<12} {'MAE (deg)':<12} {'Samples':<10}")
    print("-" * 50)

    for snr_val in sorted(unique_snrs):
        mask = (all_snrs == snr_val)
        err = all_pred_angles[mask] - all_true_angles[mask]
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(np.abs(err))
        n = np.sum(mask)
        print(f"{snr_val:<10} {rmse:<12.4f} {mae:<12.4f} {n:<10}")

    print("\nTest completed.")

def test_model_doa_scatter():
    """
    测试 -2 dB 和 0 dB 下的 DOA 估计，分别保存散点图
    """
    # ================= 路径 =================
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/dataset/M8newdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/comparemodel/DOA/CNN/best_model_doa.pth'
    save_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0710/scatter/CNN'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ================= DOA grid =================
    angle_min, angle_max, step = -60, 60, 1
    num_classes = int((angle_max - angle_min) / step) + 1
    DOA_grid = np.linspace(angle_min, angle_max, num_classes)

    # ================= 数据集 =================
    print("Loading test dataset...")
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True,
        return_spectrum=False,
        return_angle=True,
        return_iq=False,
        return_mod=False,
        return_snr=True,
        iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # ================= 模型 =================
    model = DOA_CNN(num_classes=num_classes, input_channels=2, input_size=8).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.\n")

    # ================= 收集所有样本 =================
    all_pred_angles = []
    all_true_angles = []
    all_snrs = []

    with torch.no_grad():
        for batch in test_loader:
            cov, true_angle, snr = batch
            cov = cov.to(device).float()

            output = model(cov)
            prob = torch.softmax(output, dim=1)
            pred_class = torch.argmax(prob, dim=1)

            pred_angle = DOA_grid[pred_class.cpu().numpy()]
            true_angle = true_angle.numpy()
            snr = snr.numpy()

            all_pred_angles.extend(pred_angle)
            all_true_angles.extend(true_angle)
            all_snrs.extend(snr)

    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)

    # ================= 分别筛选 -2 dB 和 0 dB 样本 =================
    snr_levels = [-2, 0]
    os.makedirs(save_dir, exist_ok=True)

    for snr_val in snr_levels:
        mask = (all_snrs == snr_val)
        pred = all_pred_angles[mask]
        true = all_true_angles[mask]
        print(f"\nSNR={snr_val} dB samples: {len(pred)}")

        if len(pred) == 0:
            print(f"No samples for SNR={snr_val} dB, skip.")
            continue

        # 绘制散点图
        plt.figure(figsize=(6, 6))
        plt.scatter(true, pred, alpha=0.5)
        min_angle, max_angle = -60, 60
        plt.plot([min_angle, max_angle], [min_angle, max_angle], 'r--')
        plt.xlabel('True DOA (deg)')
        plt.ylabel('Predicted DOA (deg)')
        # plt.title(f'DOA Scatter Plot at {snr_val} dB (CNN baseline)')
        plt.grid(True)

        # 保存文件
        save_name = f'scatter_{snr_val}dB_CNN.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Scatter plot for {snr_val} dB saved to: {save_path}")

    print("\nAll scatter plots generated.")

if __name__ == "__main__":
    main_doa()
    # test_model_doa()
    # test_model_doa_scatter()