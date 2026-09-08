import os
import argparse

parser = argparse.ArgumentParser(description="DRCDNN")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
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

#====导入数据集的数据包
class H5MultiTaskDataset(Dataset):
    """
    读取由 Matlab 保存的 data.mat 文件（格式：-v7.3)
    支持返回协方差、MUSIC谱、角度、IQ、调制类型、SNR、载频等。
    
    Matlab 保存变量约定（必须一致）：
        R_features    : (N, M, M, C)  实部+虚部+相位
        DOA_labels    : (N, num_angles)  MUSIC谱
        DOA_true      : (N,)           真实角度
        Mod_labels    : (N,)           调制类型标签 (0~11)
        TimeIQ_real   : (N, M, snapshot)  所有阵元实部
        TimeIQ_imag   : (N, M, snapshot)  所有阵元虚部
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
            iq_shape (str): '2D' 返回 (2, snapshot, M)，'complex' 返回复数张量
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
                # ---------- 读取各变量（保持与您的原始转置一致） ----------
                R = np.array(f['R_features']).astype(np.float32)          # (C, M, M, N)
                R = np.transpose(R, (3, 1, 2, 0))                         # (N, M, M, C)

                spec = np.array(f['DOA_labels']).astype(np.float32)       # (num_angles, N)
                spec = spec.T                                             # (N, num_angles)

                angle = np.array(f['DOA_true']).flatten().astype(np.float32)  # (N,)

                mod = np.array(f['Mod_labels']).flatten().astype(np.int64)    # (N,)

                iq_real = np.array(f['TimeIQ_real']).astype(np.float32)   # (snapshot, M, N)
                iq_imag = np.array(f['TimeIQ_imag']).astype(np.float32)   # (snapshot, M, N)
                iq_real = np.transpose(iq_real, (2, 1, 0))                # (N, M, snapshot)
                iq_imag = np.transpose(iq_imag, (2, 1, 0))                # (N, M, snapshot)

                # 载频（兼容旧版本无 fc_labels）
                if 'fc_labels' in f:
                    fc = np.array(f['fc_labels']).flatten().astype(np.float32)  # (N,)
                else:
                    fc = np.zeros(R.shape[0], dtype=np.float32)
                    print(f"Warning: {snr_dir} does not contain 'fc_labels'.")

                n = R.shape[0]   # 样本数
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
            self.cov = np.concatenate(cov_list, axis=0)          # (N_total, M, M, C)
            self.spectrum = np.concatenate(spectrum_list, axis=0)
            self.angle = np.concatenate(angle_list, axis=0)
            self.mod = np.concatenate(mod_list, axis=0)
            self.iq_real = np.concatenate(iq_real_list, axis=0)  # (N_total, M, snapshot)
            self.iq_imag = np.concatenate(iq_imag_list, axis=0)
            self.snr = np.concatenate(snr_list_vals, axis=0)
            self.fc = np.concatenate(fc_list, axis=0)            # 原始载频（Hz）
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
            print(f"IQ shape: {self.iq_real.shape} (samples, M, snapshot)")

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
            cov = self.cov[idx]                     # (M, M, C)
            cov = np.transpose(cov, (2, 0, 1))      # (C, M, M)  通道在前
            ret.append(torch.tensor(cov, dtype=torch.float32))
        if self.return_spectrum:
            ret.append(torch.tensor(self.spectrum[idx], dtype=torch.float32))
        if self.return_angle:
            ret.append(torch.tensor(self.angle[idx], dtype=torch.float32))
        if self.return_iq:
            # ---------- 关键修改：返回形状 (2, snapshot, M) ----------
            if self.iq_shape == '2D':
                # self.iq_real[idx] 形状为 (M, snapshot)
                iq_real = self.iq_real[idx]   # (M, snapshot)
                iq_imag = self.iq_imag[idx]   # (M, snapshot)
                # 堆叠成 (2, M, snapshot)
                iq = np.stack([iq_real, iq_imag], axis=0)
                # 转置为 (2, snapshot, M)
                iq = iq.transpose(0, 2, 1)
                ret.append(torch.tensor(iq, dtype=torch.float32))
            else:   # 'complex'
                iq_complex = self.iq_real[idx] + 1j * self.iq_imag[idx]  # (M, snapshot)
                # 转置为 (snapshot, M)
                if iq_complex.ndim == 2:
                    iq_complex = iq_complex.T
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

# ==================== 2. findpeaks 函数 ====================
def findpeaks(x, K, DOA):
    x = np.array(x)
    n = len(x)
    indexes, _ = scipy.signal.find_peaks(x, distance=1)
    # ... 原有代码 ...
    for i in range(K):
        idx = top_k_indexes[i]
        # ---------- 修复边界插值 ----------
        if idx == 0:
            if n == 1:
                p[i] = DOA[idx]
            else:
                # 确保分母不为零
                denom = x[idx] + x[idx+1]
                if abs(denom) < 1e-12:
                    p[i] = DOA[idx]  # 直接取边界网格值
                else:
                    p[i] = (x[idx] * DOA[idx] + x[idx+1] * DOA[idx+1]) / denom
        elif idx == n - 1:
            denom = x[idx-1] + x[idx]
            if abs(denom) < 1e-12:
                p[i] = DOA[idx]
            else:
                p[i] = (x[idx-1] * DOA[idx-1] + x[idx] * DOA[idx]) / denom
        else:
            left_val = x[idx-1]
            right_val = x[idx+1]
            current_val = x[idx]
            if right_val > left_val:
                denom = right_val + current_val
                if abs(denom) < 1e-12:
                    p[i] = DOA[idx]
                else:
                    p[i] = (right_val * DOA[idx+1] + current_val * DOA[idx]) / denom
            else:
                denom = left_val + current_val
                if abs(denom) < 1e-12:
                    p[i] = DOA[idx]
                else:
                    p[i] = (left_val * DOA[idx-1] + current_val * DOA[idx]) / denom
    return p, DOA[ind]


# ==================== 3. 模型组件 ====================
class DSConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DSConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CRBlock(nn.Module):
    def __init__(self, channels):
        super(CRBlock, self).__init__()
        self.dsconv1 = DSConv1d(channels, channels)
        self.dsconv2 = DSConv1d(channels, channels)

    def forward(self, x):
        identity = x
        x1 = self.dsconv1(x)
        x2 = self.dsconv2(x1)
        return x2 + identity

class CRDCNN(nn.Module):
    def __init__(self):
        super(CRDCNN, self).__init__()
        self.input_conv = nn.Conv1d(16, 64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.cr_blocks = nn.ModuleList([CRBlock(64) for _ in range(6)])
        self.output_conv = nn.Conv1d(64, 128, kernel_size=1)

    def forward(self, x):
        B, C, T, M = x.shape
        x = x.permute(0, 1, 3, 2)          # [B, 2, M, T]
        x = x.reshape(B, C * M, T)         # [B, 16, T]
        x = self.input_conv(x)
        x = self.bn(x)
        x = self.act(x)
        features = []
        for block in self.cr_blocks:
            x = block(x)
            features.append(x)
        out = self.output_conv(x)
        return features, out

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.dim_reduce = nn.Conv1d(384, 128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 121)
        )

    def forward(self, features):
        x = torch.cat(features, dim=1)     # [B, 384, T]
        x = self.dim_reduce(x)             # [B, 128, T]
        x = self.bn(x)
        x = x.permute(0, 2, 1)             # [B, T, 128]
        x, _ = self.lstm(x)                # [B, T, 128]
        x = torch.mean(x, dim=1)           # [B, 128]
        return self.fc(x)

class RobustDOANet(nn.Module):
    def __init__(self):
        super(RobustDOANet, self).__init__()
        self.crdcnn = CRDCNN()
        self.fusion = FeatureFusion()

    def forward(self, x):
        features, _ = self.crdcnn(x)
        return self.fusion(features)


# ==================== 4. 损失函数 (Focal + Dice) ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()

def combined_loss(logits, targets, alpha=0.5, beta=0.5, gamma_focal=2.0):
    focal = FocalLoss(alpha=0.25, gamma=gamma_focal, reduction='mean')
    dice = DiceLoss(smooth=1e-6)
    return alpha * focal(logits, targets) + beta * dice(logits, targets)


# ==================== 5. 训练/验证/测试函数 ====================
def train_epoch_doa(model, loader, optimizer, device, DOA_grid,
                    alpha=0.6, beta=0.4, gamma_focal=2.0):
    model.train()
    total_loss = 0
    total_rmse = 0
    total_samples = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    for spectrum, true_angle, iq, snr in pbar:
        iq = iq.to(device)
        spectrum = spectrum.to(device)
        pred = model(iq)
        loss = combined_loss(pred, spectrum, alpha, beta, gamma_focal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * iq.size(0)
        total_samples += iq.size(0)
        # RMSE
        pred_np = torch.sigmoid(pred).detach().cpu().numpy()
        true_angle_np = true_angle.cpu().numpy()
        for i in range(pred_np.shape[0]):
            angles, _ = findpeaks(pred_np[i], K=1, DOA=DOA_grid)
            pred_angle = angles[0]
            total_rmse += (pred_angle - true_angle_np[i]) ** 2
        pbar.set_postfix(loss=loss.item())
    return total_loss / total_samples, np.sqrt(total_rmse / total_samples)

def valid_epoch_doa(model, loader, device, DOA_grid,
                    alpha=0.6, beta=0.4, gamma_focal=2.0):
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_samples = 0
    with torch.no_grad():
        for spectrum, true_angle, iq, snr in loader:
            iq = iq.to(device)
            spectrum = spectrum.to(device)
            pred = model(iq)
            loss = combined_loss(pred, spectrum, alpha, beta, gamma_focal)
            total_loss += loss.item() * iq.size(0)
            total_samples += iq.size(0)
            pred_np = torch.sigmoid(pred).cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(pred_np.shape[0]):
                angles, _ = findpeaks(pred_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                total_rmse += (pred_angle - true_angle_np[i]) ** 2
    return total_loss / total_samples, np.sqrt(total_rmse / total_samples)

def test_epoch_doa(model, loader, device, DOA_grid,
                   alpha=0.6, beta=0.4, gamma_focal=2.0):
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_mae = 0
    total_samples = 0
    with torch.no_grad():
        for spectrum, true_angle, iq, snr in loader:
            iq = iq.to(device)
            spectrum = spectrum.to(device)
            pred = model(iq)
            loss = combined_loss(pred, spectrum, alpha, beta, gamma_focal)
            total_loss += loss.item() * iq.size(0)
            total_samples += iq.size(0)
            pred_np = torch.sigmoid(pred).cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(pred_np.shape[0]):
                angles, _ = findpeaks(pred_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                err = pred_angle - true_angle_np[i]
                total_rmse += err ** 2
                total_mae += abs(err)
    return total_loss / total_samples, np.sqrt(total_rmse / total_samples), total_mae / total_samples

def plot_training_curves_doa(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_rmse'], label='Train RMSE (deg)')
    plt.plot(epochs, history['val_rmse'], label='Val RMSE (deg)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (deg)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


# ==================== 6. 主函数 ====================
def main_doa():
    # ---------- 参数配置 ----------
    result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/DOA/DRCNN'
    os.makedirs(result_dir, exist_ok=True)
    model_save_path = os.path.join(result_dir, 'best_model_doa.pth')

    file_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/CNN_dataset'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 32          # 如果显存不足，可减小到16或8
    num_epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    patience = 5

    angle_min, angle_max, step = -60, 60, 1
    num_classes = int((angle_max - angle_min) / step) + 1
    DOA_grid_np = np.linspace(angle_min, angle_max, num_classes)

    # ---------- 加载数据集 ----------
    dataset = H5MultiTaskDataset(
        folder_path=file_path,
        return_cov=False,
        return_spectrum=True,
        return_angle=True,
        return_iq=True,
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

    # 注意：num_workers 如果导致卡顿，可设为 0 或 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- 模型 ----------
    model = RobustDOANet().to(device)

    model.eval()  # 设置为评估模式
    dummy_input = torch.randn(1, 2, 1024, 8).to(device)

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


    # # 计算 FLOPs 和参数量
    # input_tensor = torch.randn(1, 2, 1024, 8).to(device)
    # try:
    #     flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    #     flops, params = clever_format([flops, params], "%.3f")
    #     print(f"FLOPs: {flops}, Params: {params}")
    # except:
    #     print("FLOPs calculation skipped.")

    # ---------- 损失权重 ----------
    alpha = 0.6      # Focal Loss 权重
    beta = 0.4       # Dice Loss 权重
    gamma_focal = 2.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'lr': []
    }

    best_val_rmse = float('inf')
    early_stop_counter = 0

    print("\n" + "=" * 50)
    print("Starting IQ_ResNet_DOA Training with Focal+Dice (FD) Loss")
    print("Snapshot=768, 121 classes, -2~10dB SNR")
    print("=" * 50)

    for epoch in range(num_epochs):
        train_loss, train_rmse = train_epoch_doa(
            model, train_loader, optimizer, device, DOA_grid_np,
            alpha, beta, gamma_focal
        )
        val_loss, val_rmse = valid_epoch_doa(
            model, val_loader, device, DOA_grid_np,
            alpha, beta, gamma_focal
        )
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.6f}, RMSE: {train_rmse:.4f}°")
        print(f"  Val   - Loss: {val_loss:.6f}, RMSE: {val_rmse:.4f}°")
        print(f"  LR: {current_lr:.2e}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': best_val_rmse,
                'val_loss': val_loss
            }, model_save_path)
            early_stop_counter = 0
            print(f"  ✓ Saved best model (Val RMSE: {best_val_rmse:.4f}°)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 保存训练历史并绘图
    with open(os.path.join(result_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_training_curves_doa(history, result_dir)

    # 加载最佳模型并测试
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Best Val RMSE: {checkpoint['val_rmse']:.4f}°")

    print("\n========== Final Test Results ==========")
    test_loss, test_rmse, test_mae = test_epoch_doa(
        model, test_loader, device, DOA_grid_np,
        alpha, beta, gamma_focal
    )
    print(f"Test Loss: {test_loss:.6f}, Test RMSE: {test_rmse:.4f}°, Test MAE: {test_mae:.4f}°")

    print(f"\nResults saved to {result_dir}")

def debug_test_model_doa_DRCNN(
    test_data_path='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/CNNtestdata',
    model_path='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/DOA/DRCNN/best_model_doa.pth',
    save_dir='/home/sp432sl/ZQ/DOA_regcition_0908/CDMT-Net_0718/comparedmodle/DOA/DRCNN',
    val_loader=None,
    batch_size=64,
    num_workers=4,
    K=1,
    plot_scatter_snr=0,
    save_csv=True,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    angle_min, angle_max, step = -60, 60, 1
    num_classes = int((angle_max - angle_min) / step) + 1
    DOA_grid_np = np.linspace(angle_min, angle_max, num_classes)

    # ========== 加载测试数据集 ==========
    print("\n===== Loading Test Dataset =====")
    dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=False,
        return_spectrum=True,
        return_angle=True,
        return_iq=True,
        return_mod=False,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'
    )
    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"Total test samples: {len(dataset)}")

    # ---------- 调试信息 1：数据属性 ----------
    print("\n----- Test Data Attributes -----")
    print(f"IQ real shape: {dataset.iq_real.shape}")
    print(f"Spectrum shape: {dataset.spectrum.shape}")
    print(f"Angle shape: {dataset.angle.shape}")
    print(f"Unique SNR values: {np.unique(dataset.snr)}")
    print(f"First sample true angle: {dataset.angle[0]}")
    print(f"First sample SNR: {dataset.snr[0]}")
    if dataset.angle.ndim != 1:
        print("WARNING: dataset.angle is not 1D, it may contain multiple angles!")

    # ========== 加载模型 ==========
    print("\n===== Loading Model =====")
    model = RobustDOANet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.")

    # ========== 若提供了验证集 ==========
    if val_loader is not None:
        print("\n===== Validating on Validation Set (should be ~0.7°) =====")
        val_rmse, val_mae = compute_metrics(model, val_loader, device, DOA_grid_np, K)
        print(f"Validation RMSE: {val_rmse:.4f}°, MAE: {val_mae:.4f}°")

    # ========== 主测试循环 ==========
    print("\n===== Running Test Inference =====")
    all_pred_angles = []
    all_true_angles = []
    all_snrs = []
    debug_sample_printed = False

    with torch.no_grad():
        for batch_idx, (spectrum, true_angle, iq, snr) in enumerate(tqdm(test_loader, desc="Testing")):
            iq = iq.to(device)
            pred = model(iq)
            pred_np = pred.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()

            for i in range(pred_np.shape[0]):
                # ---------- 调试信息：第一个样本（只打印，不做预测） ----------
                if not debug_sample_printed:
                    debug_sample_printed = True
                    print("\n----- Debug: First Sample -----")
                    pred_spectrum = torch.sigmoid(pred[i]).cpu().numpy()
                    true_spectrum = spectrum[i].cpu().numpy()
                    true_ang = true_angle_np[i]
                    print(f"True DOA: {true_ang}°")
                    print(f"Pred spectrum min: {pred_spectrum.min():.4f}, max: {pred_spectrum.max():.4f}")
                    argmax_idx = np.argmax(pred_spectrum)
                    print(f"Argmax angle: {DOA_grid_np[argmax_idx]:.2f}° (value {pred_spectrum[argmax_idx]:.4f})")
                    # ⚠️ 此处暂时不调用 findpeaks，直接用 argmax 角度显示
                    pred_angle_debug = DOA_grid_np[argmax_idx]
                    print(f"Predicted angle (argmax): {pred_angle_debug}°")
                    # 绘制频谱对比图（使用 argmax 角度）
                    plt.figure(figsize=(10,4))
                    plt.plot(DOA_grid_np, true_spectrum, label='True Spectrum')
                    plt.plot(DOA_grid_np, pred_spectrum, label='Pred Spectrum')
                    plt.axvline(true_ang, color='r', linestyle='--', label='True DOA')
                    plt.axvline(pred_angle_debug, color='g', linestyle='--', label='Pred DOA (argmax)')
                    plt.xlabel('Angle (deg)')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.title('Spectrum Comparison (First Sample)')
                    plt.grid(True)
                    debug_fig_path = os.path.join(save_dir, 'debug_spectrum_first_sample.png')
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(debug_fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Debug spectrum figure saved to {debug_fig_path}")

                # ---------- 预测（直接使用 argmax） ----------
                argmax_idx = np.argmax(pred_np[i])
                pred_angle = DOA_grid_np[argmax_idx]   # ✅ 直接取网格值，不使用 findpeaks

                # ---------- 异常值检测 ----------
                if abs(pred_angle) > 100:
                    print(f"WARNING: Abnormal pred angle = {pred_angle:.2f}°, true={true_angle_np[i]:.2f}°, SNR={snr[i].item()}")

                all_pred_angles.append(pred_angle)
                all_true_angles.append(true_angle_np[i])
                all_snrs.append(snr[i].item())

    # ---------- 后续统计和绘图（与之前完全相同） ----------
    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)

    errors = all_pred_angles - all_true_angles
    overall_rmse = np.sqrt(np.mean(errors ** 2))
    overall_mae = np.mean(np.abs(errors))
    overall_max_err = np.max(np.abs(errors))
    overall_std = np.std(errors)

    print("\n========== Overall Performance ==========")
    print(f"RMSE: {overall_rmse:.4f}°")
    print(f"MAE : {overall_mae:.4f}°")
    print(f"Max Err: {overall_max_err:.4f}°")
    print(f"Std Dev: {overall_std:.4f}°\n")

    unique_snrs = sorted(np.unique(all_snrs))
    snr_stats = []
    print("========== SNR-wise Performance ==========")
    print(f"{'SNR (dB)':<10} {'RMSE (deg)':<12} {'MAE (deg)':<12} {'Max Err':<12} {'Samples':<10}")
    print("-" * 70)
    for snr_val in unique_snrs:
        mask = (all_snrs == snr_val)
        err_snr = errors[mask]
        rmse = np.sqrt(np.mean(err_snr ** 2))
        mae = np.mean(np.abs(err_snr))
        max_err = np.max(np.abs(err_snr))
        n = np.sum(mask)
        snr_stats.append({
            'SNR': snr_val,
            'RMSE': rmse,
            'MAE': mae,
            'MaxErr': max_err,
            'Samples': n
        })
        print(f"{snr_val:<10} {rmse:<12.4f} {mae:<12.4f} {max_err:<12.4f} {n:<10}")

    if save_csv:
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(snr_stats)
        csv_path = os.path.join(save_dir, 'test_results_snr.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSNR-wise results saved to {csv_path}")

    # 绘制箱线图、直方图、散点图、误差 vs SNR 曲线（代码与之前相同，不再重复，直接省略以节省空间）
    # 您可以将之前的绘图代码复制过来（从 # 箱线图 开始到 # 返回结果）

    # 由于篇幅，这里只返回结果，绘图部分您可自行补全（或直接使用之前的代码）
    results = {
        'overall': {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'max_err': overall_max_err,
            'std': overall_std
        },
        'snr_wise': snr_stats,
        'errors': errors,
        'snrs': all_snrs,
        'pred_angles': all_pred_angles,
        'true_angles': all_true_angles
    }
    return results


def save_iq_resnet_scatter_by_snr(target_snr, save_filename, plot_title=None):
    """
    保存 IQ_ResNet 在指定 SNR 下的 DOA 散点图
    
    Args:
        target_snr: 目标信噪比（整数，如 0, -2）
        save_filename: 保存的文件名（如 'scatter_0dB_IQ_ResNet.png'）
        plot_title: 图片标题，默认自动生成 "IQ_ResNet DOA Scatter Plot at X dB"
    """
    # ========== 固定路径 ==========
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/dataset/test_modulation_TP_net-1616'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/comparedmodel/DOA/IQ_ResNet/best_model_doa.pth'
    save_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/scatter/IQ_ResNET'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DOA 网格
    angle_min, angle_max, step = -60, 60, 1
    num_classes = int((angle_max - angle_min) / step) + 1
    DOA_grid_np = np.linspace(angle_min, angle_max, num_classes)
    DOA_grid = torch.tensor(DOA_grid_np, dtype=torch.float32, device=device)

    # 数据集
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=False,
        return_spectrum=False,
        return_angle=True,
        return_iq=True,
        return_mod=False,
        return_snr=True,
        return_fc=False,
        iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 模型
    model = IQ_ResNet_DOA(num_classes_doa=num_classes, in_ch=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.\n")

    # 收集指定 SNR 的样本
    pred_angles = []
    true_angles = []

    with torch.no_grad():
        for batch in test_loader:
            true_angle, iq, snr = batch

            # IQ 形状处理
            if iq.dim() == 4:
                if iq.shape[1] != 2 and iq.shape[3] == 2:
                    iq = iq.permute(0, 3, 1, 2)
            elif iq.dim() == 3:
                B, D1, D2 = iq.shape
                if D1 == 768 and D2 == 8:
                    iq = iq.unsqueeze(1).repeat(1, 2, 1, 1)
                elif D1 == 8 and D2 == 768:
                    iq = iq.permute(0, 2, 1).unsqueeze(1).repeat(1, 2, 1, 1)
                else:
                    continue  # 形状异常则跳过该 batch
            else:
                continue

            iq = iq.float().to(device)
            logits = model(iq)
            prob = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(prob, dim=1)
            pred_angle = DOA_grid[pred_class].cpu().numpy()
            true_angle_np = true_angle.numpy()
            snr_np = snr.numpy()

            mask = (snr_np == target_snr)
            if np.any(mask):
                pred_angles.extend(pred_angle[mask])
                true_angles.extend(true_angle_np[mask])

    if len(pred_angles) == 0:
        print(f"No samples found at SNR = {target_snr} dB.")
        return

    # 画图
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(true_angles, pred_angles, alpha=0.5, s=10)
    plt.plot([-60, 60], [-60, 60], 'r--', linewidth=1.5)
    plt.xlabel('True DOA (deg)')
    plt.ylabel('Predicted DOA (deg)')
    if plot_title is None:
        plt.title(f'IQ_ResNet DOA Scatter Plot at {target_snr} dB')
    else:
        plt.title(plot_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{target_snr} dB scatter plot saved to: {save_path}")


def test_iq_resnet_scatter():
    """生成 0 dB 和 -2 dB 的散点图"""
    save_iq_resnet_scatter_by_snr(0, 'scatter_0dB_IQ_ResNet.png')
    save_iq_resnet_scatter_by_snr(-2, 'scatter_-2dB_IQ_ResNet.png')

if __name__ == "__main__":
    main_doa()
    # debug_test_model_doa_DRCNN()
    # test_model_doa_iq_resnet()
    # test_iq_resnet_scatter()