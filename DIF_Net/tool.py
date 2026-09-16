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
import scipy.io as sio

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

# 只导入理想的窄带信号协方差 标签为真实的方位角值
import os
import numpy as np
import h5py  # 改用 h5py
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

# ====波束估计 数据集导入函数======
class H5DatasetSpectrum(Dataset):
    def __init__(self, folder_path, snr_list=None, transform=None):
        """
        数据集包含：
            cov: 协方差矩阵特征 (2, M, M)
            spectrum_label: 谱标签 (num_classes,)，例如 MUSIC 谱或 multi-hot
            true_angle: 真实角度 (1,)
            snr: SNR 值 (int)
        """
        self.R_features_list = []
        self.labels_list = []        # 谱标签
        self.true_angles_list = []   # 真实角度
        self.snr_labels_list = []
        self.transform = transform

        # 获取所有 SNR 子文件夹
        snr_folders = [f for f in os.listdir(folder_path) if f.startswith('SNR_')]
        if snr_list is not None:
            snr_folders = [f for f in snr_folders 
                          if int(f.split('_')[-1].replace('dB', '')) in snr_list]

        for snr_dir in sorted(snr_folders):
            snr_value = int(snr_dir.split('_')[-1].replace('dB', ''))
            mat_file = os.path.join(folder_path, snr_dir, 'data.mat')
            with h5py.File(mat_file, 'r') as f:
                # 协方差特征(2, M, M, samples) -> (samples, M, M, 2)
                R_raw = np.array(f['R_features'])
                R = np.transpose(R_raw, (3, 1, 2, 0))
                spectrum = np.array(f['DOA_labels_music'])   # 假设为 (samples, num_classes)
                if spectrum.ndim == 2 and spectrum.shape[0] != R.shape[0]:
                    spectrum = spectrum.T
                # 真实角度：形状 (samples, 1) 或 (samples,)
                true_angle = np.array(f['DOA_true']).flatten()   # 展平为 (samples,)
                
            self.R_features_list.append(R.astype(np.float32))
            self.labels_list.append(spectrum.astype(np.float32))
            self.true_angles_list.append(true_angle.astype(np.float32))
            self.snr_labels_list.append(np.full(R.shape[0], snr_value, dtype=np.int32))

        # 合并所有数据
        self.R_features = np.concatenate(self.R_features_list, axis=0)      # (N, M, M, 2)
        self.labels = np.concatenate(self.labels_list, axis=0)              # (N, num_classes)
        self.true_angles = np.concatenate(self.true_angles_list, axis=0)    # (N,)
        self.SNR_labels = np.concatenate(self.snr_labels_list, axis=0)      # (N,)

    def __len__(self):
        return len(self.R_features)

    def __getitem__(self, idx):
        cov = self.R_features[idx]                     # (M, M, 2)
        cov = np.transpose(cov, (2, 0, 1))             # (2, M, M)
        spectrum = self.labels[idx]                    # (num_classes,)
        true_angle = self.true_angles[idx]             # scalar
        snr = self.SNR_labels[idx]
        
        if self.transform:
            cov = self.transform(cov)
        
        # 返回四个值：协方差、谱标签、真实角度、SNR
        return (
            torch.tensor(cov, dtype=torch.float32),
            torch.tensor(spectrum, dtype=torch.float32),
            torch.tensor(true_angle, dtype=torch.float32),
            torch.tensor(snr, dtype=torch.long)
        )

# ====波束估计和调制识别任务 数据集导入函数======
class H5MultiTaskDataset(Dataset):
    def __init__(self, folder_path, snr_list=None,
                 return_cov=True, return_spectrum=True, return_angle=True,
                 return_iq=True, return_mod=True, return_snr=True,
                 return_fc=False, fc_norm='minmax', iq_shape='2D'):
        """
        fc_norm: 归一化方式
            - 'minmax'   : 线性缩放到 [0,1]
            - 'logminmax': 先取 log10(fc)，再线性缩放到 [0,1]（适用于宽频带回归）
            - 'zscore'   : 标准化 (fc - mean)/std
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

        self.cov = None
        self.spectrum = None
        self.angle = None
        self.iq_real = None
        self.iq_imag = None
        self.mod = None
        self.snr = None
        self.fc = None          # 存储归一化前的原始值（或归一化后的值，取决于用途）

        # 归一化参数
        self.fc_min = None
        self.fc_max = None
        self.fc_mean = None
        self.fc_std = None

        # 获取所有SNR子文件夹
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
                # 协方差特征
                R_raw = np.array(f['R_features']).astype(np.float32)
                R = np.transpose(R_raw, (3, 1, 2, 0))      # (samples, M, M, 2)
                # MUSIC谱
                spec = np.array(f['DOA_labels_music']).astype(np.float32)
                if spec.ndim == 2 and spec.shape[0] != R.shape[0]:
                    spec = spec.T
                angle = np.array(f['DOA_true']).flatten().astype(np.float32)
                mod = np.array(f['Mod_labels']).flatten().astype(np.int64)
                iq_real = np.array(f['TimeIQ_real']).astype(np.float32).T
                iq_imag = np.array(f['TimeIQ_imag']).astype(np.float32).T
                # 读取载频
                if 'fc_labels' in f:
                    fc = np.array(f['fc_labels']).flatten().astype(np.float32)
                else:
                    fc = np.zeros(R.shape[0], dtype=np.float32)
                    print(f"Warning: {snr_dir} does not contain 'fc_labels'.")

                n = R.shape[0]
                assert spec.shape[0] == n
                assert angle.shape[0] == n
                assert mod.shape[0] == n
                assert iq_real.shape[0] == n
                assert fc.shape[0] == n

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

        # ---------- 载频归一化处理 ----------
        if self.return_fc and self.fc_norm is not None:
            if self.fc_norm == 'minmax':
                self.fc_min = self.fc.min()
                self.fc_max = self.fc.max()
                if self.fc_max - self.fc_min < 1e-6:
                    self.fc_min = 0
                    self.fc_max = 1
                self.fc = (self.fc - self.fc_min) / (self.fc_max - self.fc_min)
                print(f"MinMax norm: fc_min={self.fc_min:.2e}, fc_max={self.fc_max:.2e}")
            elif self.fc_norm == 'logminmax':
                # 对数归一化：先取 log10(fc) -> 再 min-max 到 [0,1]
                fc_log = np.log10(self.fc + 1e-12)   # 加极小量防止 log10(0)
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
        if self.return_fc:
            print(f"FC labels shape: {self.fc.shape} (samples,)")

    def get_fc_denorm(self, fc_normed):
        """将归一化后的载频转换回原始值（NumPy 版本，用于 RMSE 等）"""
        if self.fc_norm == 'minmax':
            return fc_normed * (self.fc_max - self.fc_min) + self.fc_min
        elif self.fc_norm == 'logminmax':
            log_fc = fc_normed * (self.fc_max - self.fc_min) + self.fc_min
            return 10 ** log_fc
        elif self.fc_norm == 'zscore':
            return fc_normed * self.fc_std + self.fc_mean
        else:
            return fc_normed   # 未归一化，直接返回

    def denorm_tensor(self, fc_normed_tensor):
        """
        张量版本的反归一化，用于训练时对模型输出的 fc_hat_norm 进行反归一化（保持计算图）
        fc_normed_tensor: torch.Tensor, 形状任意
        """
        if self.fc_norm == 'minmax':
            # 注意：训练时 minmax 建议不要 clamp，因为模型输出可能超出 [0,1] 但反向传播仍可优化
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
            cov = self.cov[idx]
            cov = np.transpose(cov, (2, 0, 1))      # (2, M, M)
            ret.append(torch.tensor(cov, dtype=torch.float32))
        if self.return_spectrum:
            ret.append(torch.tensor(self.spectrum[idx], dtype=torch.float32))
        if self.return_angle:
            ret.append(torch.tensor(self.angle[idx], dtype=torch.float32))
        if self.return_iq:
            if self.iq_shape == '2D':
                iq = np.stack([self.iq_real[idx], self.iq_imag[idx]], axis=0)  # (2, snapshot)
                iq = iq[..., np.newaxis]
                ret.append(torch.tensor(iq, dtype=torch.float32))
            else:
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

# class H5MultiTaskDataset(Dataset):
#     def __init__(self, folder_path, snr_list=None,
#                  return_cov=True, return_spectrum=True, return_angle=True,
#                  return_iq=True, return_mod=True, return_snr=True,
#                  return_fc=False, fc_norm='minmax', iq_shape='2D'):
#         """
#         fc_norm: 归一化方式
#             - 'minmax'   : 线性缩放到 [0,1]
#             - 'logminmax': 先取 log10(fc)，再线性缩放到 [0,1]（适用于宽频带回归）
#             - 'zscore'   : 标准化 (fc - mean)/std
#         """
#         self.return_cov = return_cov
#         self.return_spectrum = return_spectrum
#         self.return_angle = return_angle
#         self.return_iq = return_iq
#         self.return_mod = return_mod
#         self.return_snr = return_snr
#         self.return_fc = return_fc
#         self.fc_norm = fc_norm
#         self.iq_shape = iq_shape

#         self.cov = None
#         self.spectrum = None
#         self.angle = None
#         self.iq_real = None
#         self.iq_imag = None
#         self.mod = None
#         self.snr = None
#         self.fc = None

#         # 归一化参数
#         self.fc_min = None
#         self.fc_max = None
#         self.fc_mean = None
#         self.fc_std = None
#         self.cov_max_abs = None      # 协方差矩阵归一化因子

#         # 获取所有SNR子文件夹
#         snr_folders = [f for f in os.listdir(folder_path) if f.startswith('SNR_')]
#         if snr_list is not None:
#             snr_folders = [f for f in snr_folders
#                            if int(f.split('_')[-1].replace('dB', '')) in snr_list]
#         snr_folders.sort()

#         cov_list, spectrum_list, angle_list = [], [], []
#         iq_real_list, iq_imag_list, mod_list, snr_list_vals, fc_list = [], [], [], [], []

#         for snr_dir in snr_folders:
#             snr_value = int(snr_dir.split('_')[-1].replace('dB', ''))
#             mat_file = os.path.join(folder_path, snr_dir, 'data.mat')
#             with h5py.File(mat_file, 'r') as f:
#                 R_raw = np.array(f['R_features']).astype(np.float32)
#                 R = np.transpose(R_raw, (3, 1, 2, 0))      # (samples, M, M, 2)
#                 spec = np.array(f['DOA_labels_music']).astype(np.float32)
#                 if spec.ndim == 2 and spec.shape[0] != R.shape[0]:
#                     spec = spec.T
#                 angle = np.array(f['DOA_true']).flatten().astype(np.float32)
#                 mod = np.array(f['Mod_labels']).flatten().astype(np.int64)
#                 iq_real = np.array(f['TimeIQ_real']).astype(np.float32).T
#                 iq_imag = np.array(f['TimeIQ_imag']).astype(np.float32).T
#                 if 'fc_labels' in f:
#                     fc = np.array(f['fc_labels']).flatten().astype(np.float32)
#                 else:
#                     fc = np.zeros(R.shape[0], dtype=np.float32)
#                     print(f"Warning: {snr_dir} does not contain 'fc_labels'.")

#                 n = R.shape[0]
#                 assert spec.shape[0] == n
#                 assert angle.shape[0] == n
#                 assert mod.shape[0] == n
#                 assert iq_real.shape[0] == n
#                 assert fc.shape[0] == n

#                 cov_list.append(R)
#                 spectrum_list.append(spec)
#                 angle_list.append(angle)
#                 mod_list.append(mod)
#                 iq_real_list.append(iq_real)
#                 iq_imag_list.append(iq_imag)
#                 fc_list.append(fc)
#                 snr_list_vals.append(np.full(n, snr_value, dtype=np.int32))

#         # 合并所有数据
#         if cov_list:
#             self.cov = np.concatenate(cov_list, axis=0)
#             self.spectrum = np.concatenate(spectrum_list, axis=0)
#             self.angle = np.concatenate(angle_list, axis=0)
#             self.mod = np.concatenate(mod_list, axis=0)
#             self.iq_real = np.concatenate(iq_real_list, axis=0)
#             self.iq_imag = np.concatenate(iq_imag_list, axis=0)
#             self.snr = np.concatenate(snr_list_vals, axis=0)
#             self.fc = np.concatenate(fc_list, axis=0)
#         else:
#             raise ValueError(f"No data found in {folder_path}")

#         # ---------- 协方差矩阵归一化（全局归一化） ----------
#         if self.return_cov:
#             # 计算所有样本中协方差矩阵绝对值的最大值
#             self.cov_max_abs = np.abs(self.cov).max()
#             # 归一化到 [-1, 1] 区间
#             self.cov = self.cov / (self.cov_max_abs + 1e-8)
#             print(f"Covariance normalization: global max abs = {self.cov_max_abs:.6f}")

#         # ---------- 载频归一化处理 ----------
#         if self.return_fc and self.fc_norm is not None:
#             if self.fc_norm == 'minmax':
#                 self.fc_min = self.fc.min()
#                 self.fc_max = self.fc.max()
#                 if self.fc_max - self.fc_min < 1e-6:
#                     self.fc_min = 0
#                     self.fc_max = 1
#                 self.fc = (self.fc - self.fc_min) / (self.fc_max - self.fc_min)
#                 print(f"MinMax norm: fc_min={self.fc_min:.2e}, fc_max={self.fc_max:.2e}")
#             elif self.fc_norm == 'logminmax':
#                 fc_log = np.log10(self.fc + 1e-12)
#                 self.fc_min = fc_log.min()
#                 self.fc_max = fc_log.max()
#                 self.fc = (fc_log - self.fc_min) / (self.fc_max - self.fc_min)
#                 print(f"LogMinMax norm: log10(fc) min={self.fc_min:.4f}, max={self.fc_max:.4f}")
#             elif self.fc_norm == 'zscore':
#                 self.fc_mean = self.fc.mean()
#                 self.fc_std = self.fc.std()
#                 self.fc = (self.fc - self.fc_mean) / self.fc_std
#                 print(f"Z-score norm: fc_mean={self.fc_mean:.2e}, fc_std={self.fc_std:.2e}")
#             else:
#                 raise ValueError(f"Unknown fc_norm: {self.fc_norm}")
#         elif self.return_fc:
#             print("FC labels are used in original scale (no normalization).")

#         self.length = len(self.cov)
#         print(f"Total samples loaded: {self.length}")
#         if self.return_iq:
#             print(f"IQ shape: {self.iq_real.shape} (samples, snapshot)")
#         if self.return_fc:
#             print(f"FC labels shape: {self.fc.shape} (samples,)")
#         if self.return_cov:
#             print(f"Covariance shape: {self.cov.shape} (samples, M, M, 2)")

#     def get_fc_denorm(self, fc_normed):
#         """将归一化后的载频转换回原始值（NumPy 版本，用于 RMSE 等）"""
#         if self.fc_norm == 'minmax':
#             return fc_normed * (self.fc_max - self.fc_min) + self.fc_min
#         elif self.fc_norm == 'logminmax':
#             log_fc = fc_normed * (self.fc_max - self.fc_min) + self.fc_min
#             return 10 ** log_fc
#         elif self.fc_norm == 'zscore':
#             return fc_normed * self.fc_std + self.fc_mean
#         else:
#             return fc_normed

#     def denorm_tensor(self, fc_normed_tensor):
#         """张量版本的反归一化，用于训练时对模型输出的 fc_hat_norm 进行反归一化（保持计算图）"""
#         if self.fc_norm == 'minmax':
#             return fc_normed_tensor * (self.fc_max - self.fc_min) + self.fc_min
#         elif self.fc_norm == 'logminmax':
#             log_fc = fc_normed_tensor * (self.fc_max - self.fc_min) + self.fc_min
#             return 10 ** log_fc
#         elif self.fc_norm == 'zscore':
#             return fc_normed_tensor * self.fc_std + self.fc_mean
#         else:
#             return fc_normed_tensor

#     def denorm_cov(self, cov_normed):
#         """如果需要将归一化的协方差矩阵还原回原始尺度（一般用于分析，MUSIC不需要）"""
#         return cov_normed * (self.cov_max_abs + 1e-8)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         ret = []
#         if self.return_cov:
#             cov = self.cov[idx]
#             cov = np.transpose(cov, (2, 0, 1))      # (2, M, M)
#             ret.append(torch.tensor(cov, dtype=torch.float32))
#         if self.return_spectrum:
#             ret.append(torch.tensor(self.spectrum[idx], dtype=torch.float32))
#         if self.return_angle:
#             ret.append(torch.tensor(self.angle[idx], dtype=torch.float32))
#         if self.return_iq:
#             if self.iq_shape == '2D':
#                 iq = np.stack([self.iq_real[idx], self.iq_imag[idx]], axis=0)  # (2, snapshot)
#                 iq = iq[..., np.newaxis]
#                 ret.append(torch.tensor(iq, dtype=torch.float32))
#             else:
#                 iq_complex = self.iq_real[idx] + 1j * self.iq_imag[idx]
#                 iq_complex = iq_complex[..., np.newaxis] if iq_complex.ndim == 1 else iq_complex
#                 ret.append(torch.tensor(iq_complex, dtype=torch.complex64))
#         if self.return_mod:
#             ret.append(torch.tensor(self.mod[idx], dtype=torch.long))
#         if self.return_snr:
#             ret.append(torch.tensor(self.snr[idx], dtype=torch.long))
#         if self.return_fc:
#             ret.append(torch.tensor(self.fc[idx], dtype=torch.float32))
#         if len(ret) == 1:
#             return ret[0]
#         return tuple(ret)


# 导向矢量的重构
def music_spectrum_numpy(R, M, d, fc_hat, angle_grid, num_signal=1):
    """
    R: 复数协方差矩阵 (M, M)
    fc_hat: 预测载频 (Hz)
    返回: 谱 (len(angle_grid),)
    """
    c = 3e8
    lambda_hat = c / (fc_hat + 1e-8)
    eigvals, eigvecs = np.linalg.eig(R)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, num_signal:]
    P = []
    for theta in angle_grid:
        theta_rad = np.deg2rad(theta)
        a = np.exp(1j * 2 * np.pi * d * np.arange(M)[:, None] * np.sin(theta_rad) / lambda_hat)
        val = 1 / np.abs(a.conj().T @ En @ En.conj().T @ a)
        P.append(val.item())
    P = np.array(P)
    # 峰值归一化（与训练标签一致）
    P = P / (np.max(P) + 1e-8)
    return P

# 导向矢量的重构
# ------------------------- 3. 可微 MUSIC 层 -------------------------
class DifferentiableMUSIC(nn.Module):
    def __init__(self, M, d, angle_grid, fc_norm_params, eps=1e-6):
        super().__init__()
        self.M = M
        self.d = d
        self.eps = eps
        self.register_buffer("angle_grid", torch.tensor(angle_grid, dtype=torch.float))
        self.fc_norm_type = fc_norm_params['type']
        self.fc_min = fc_norm_params.get('min', None)
        self.fc_max = fc_norm_params.get('max', None)
        self.fc_mean = fc_norm_params.get('mean', None)
        self.fc_std = fc_norm_params.get('std', None)

        # 注册 buffer 以便模型保存时一并存储
        if self.fc_min is not None:
            self.register_buffer('fc_min_t', torch.tensor(self.fc_min))
            self.register_buffer('fc_max_t', torch.tensor(self.fc_max))
        if self.fc_mean is not None:
            self.register_buffer('fc_mean_t', torch.tensor(self.fc_mean))
            self.register_buffer('fc_std_t', torch.tensor(self.fc_std))

    def denorm_fc(self, fc_normed):
        if self.fc_norm_type == 'minmax':
            return fc_normed * (self.fc_max - self.fc_min) + self.fc_min
        elif self.fc_norm_type == 'logminmax':
            log_fc = fc_normed * (self.fc_max - self.fc_min) + self.fc_min
            return 10 ** log_fc
        elif self.fc_norm_type == 'zscore':
            return fc_normed * self.fc_std + self.fc_mean
        else:
            return fc_normed

    def forward(self, R, fc_hat_norm):
        B, M, _ = R.shape
        fc_hat = self.denorm_fc(fc_hat_norm)   # (B,)
        c = 3e8
        lambda_hat = c / (fc_hat + 1e-8)

        R_reg = R + self.eps * torch.eye(M, dtype=R.dtype, device=R.device)
        eigvals, eigvecs = torch.linalg.eigh(R_reg)
        idx = torch.argsort(eigvals, dim=-1, descending=True)
        eigvecs = torch.gather(eigvecs, dim=2, index=idx.unsqueeze(1).expand(-1, M, -1))
        En = eigvecs[:, :, 1:]                     # (B, M, M-1)

        sin_theta = torch.sin(torch.deg2rad(self.angle_grid))  # (A,)
        m = torch.arange(M, device=R.device).float().view(1, M, 1)
        a = torch.exp(1j * 2 * torch.pi * self.d * m * sin_theta.view(1, 1, -1) / lambda_hat.view(B, 1, 1))

        EnEnH = En @ En.conj().transpose(-1, -2)
        a_conj = a.conj()
        EnEnH_a = EnEnH @ a
        val = torch.einsum('bma,bma->ba', a_conj, EnEnH_a)
        P = 1.0 / (torch.abs(val) + 1e-8)
        P = P / (P.max(dim=1, keepdim=True)[0] + 1e-8)
        return P

class DOAMusicLayer(nn.Module):
    """包装 MUSIC 层，将输入的 (B,2,M,M) 转换为复数矩阵"""
    def __init__(self, M, d, angle_grid, fc_norm_params, eps=1e-6):
        super().__init__()
        self.music = DifferentiableMUSIC(M, d, angle_grid, fc_norm_params, eps)

    def forward(self, cov_real_imag, fc_hat_norm):
        # cov_real_imag: (B, 2, M, M)
        R = torch.complex(cov_real_imag[:, 0, :, :], cov_real_imag[:, 1, :, :])
        return self.music(R, fc_hat_norm)


def get_spec_weight(snr_db, train_min_snr=-2, train_max_snr=10, min_weight=0.0, max_weight=0.15):
    """
    基于 SNR 自适应计算谱损失权重
    Args:
        snr_db: 当前 batch 的平均 SNR (dB)
        train_min_snr: 训练集中最小 SNR
        train_max_snr: 训练集中最大 SNR
        min_weight: 最低权重（对应 <= min_snr）
        max_weight: 最高权重（对应 >= max_snr）
    Returns:
        weight: 当前 batch 的谱损失权重
    """
    if snr_db <= train_min_snr:
        return min_weight
    elif snr_db >= train_max_snr:
        return max_weight
    else:
        # 线性插值
        return min_weight + (max_weight - min_weight) * (snr_db - train_min_snr) / (train_max_snr - train_min_snr)

# class H5DatasetMultiSNR(Dataset):
#     def __init__(self, folder_path, snr_list=None, transform=None):
#         """
#         参数:
#             folder_path: 数据集根目录（包含SNR_*子文件夹）
#             snr_list: 可选，指定加载哪些SNR，如[-12, -8, 0, 10]
#             transform: 可选的数据变换
#         """
#         self.R_features_list = []
#         self.DOA_labels_list = []
#         self.snr_labels_list = []
#         self.transform = transform

#         # 获取所有SNR子文件夹
#         snr_folders = [f for f in os.listdir(folder_path) if f.startswith('SNR_')]
        
#         # 如果指定了snr_list，只加载指定的SNR
#         if snr_list is not None:
#             snr_folders = [f for f in snr_folders 
#                           if int(f.split('_')[-1]) in snr_list]

#         print(f"加载 {len(snr_folders)} 个SNR文件夹...")

#         for snr_dir in sorted(snr_folders):
#             snr_value = int(snr_dir.split('_')[-1].replace('dB', ''))
#             mat_file = os.path.join(folder_path, snr_dir, 'data.mat')
            
#             if not os.path.exists(mat_file):
#                 print(f"警告: {mat_file} 不存在，跳过")
#                 continue
            
#             try:
#                 # 加载.mat文件
#                 data = sio.loadmat(mat_file)
                
#                 # 提取特征和标签
#                 # R_features: [samples, M, M, 2] (实部,虚部)
#                 R_features = data['R_features']
#                 DOA_labels = data['DOA_labels'].flatten()
                
#                 # 打印调试信息
#                 print(f"  SNR {snr_value}dB: {len(R_features)} 个样本")
                
#                 self.R_features_list.append(R_features.astype(np.float32))
#                 self.DOA_labels_list.append(DOA_labels.astype(np.float32))
#                 self.snr_labels_list.append(np.full(len(R_features), snr_value, dtype=np.int32))
                
#             except Exception as e:
#                 print(f"错误加载 {snr_dir}: {e}")
#                 continue

#         # 合并所有数据
#         if len(self.R_features_list) == 0:
#             raise ValueError(f"在 {folder_path} 中没有找到有效数据")
        
#         self.R_features = np.concatenate(self.R_features_list, axis=0)
#         self.DOA_labels = np.concatenate(self.DOA_labels_list, axis=0)
#         self.SNR_labels = np.concatenate(self.snr_labels_list, axis=0)
        
#         print(f"\n总计: {len(self.R_features)} 个样本")
#         print(f"特征维度: {self.R_features.shape}")
#         print(f"DOA范围: [{self.DOA_labels.min():.2f}, {self.DOA_labels.max():.2f}]")
#         print(f"SNR范围: {np.unique(self.SNR_labels)}")

#     def __len__(self):
#         return len(self.R_features)

#     def __getitem__(self, idx):
#         """
#         返回:
#             cov_features: 协方差特征 (2, M, M) - 2通道:实部、虚部
#             doa_label: DOA标签 (度)
#             snr_label: SNR标签 (dB)
#         """
#         # 获取协方差矩阵 [M, M, 2]
#         cov = self.R_features[idx]
        
#         # 转换为 [2, M, M] 格式（PyTorch Conv2D需要）
#         # 原格式: [M, M, 2] -> 转置为 [2, M, M]
#         cov_features = np.transpose(cov, (2, 0, 1))
        
#         doa_label = self.DOA_labels[idx]
#         snr_label = self.SNR_labels[idx]
        
#         # 可选数据变换
#         if self.transform:
#             cov_features = self.transform(cov_features)
        
#         return (
#             torch.tensor(cov_features, dtype=torch.float32),
#             torch.tensor(doa_label, dtype=torch.float32),
#             torch.tensor(snr_label, dtype=torch.long)
#         )

# 只导入理想的窄带信号协方差 标签为谱方位角
# ---------- 按 SNR 评估 ----------
def evaluate_by_snr(model, dataloader, device, DOA_grid, num_sources, result_dir):
    model.eval()
    snr_results = defaultdict(lambda: {'errors': [], 'abs_errors': []})
    with torch.no_grad():
        for cov, spectrum, true_angle, snr in dataloader:
            cov = cov.to(device)
            spectrum = spectrum.to(device)
            pred_spec = model(cov)
            pred_spec_np = pred_spec.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            snr_np = snr.cpu().numpy()
            for i in range(pred_spec_np.shape[0]):
                angles, _ = findpeaks(pred_spec_np[i], K=num_sources, DOA=DOA_grid)
                pred_angle = angles[0] if len(angles) > 0 else np.nan
                true_angle_val = true_angle_np[i]
                if not np.isnan(pred_angle):
                    err = pred_angle - true_angle_val
                    snr_results[snr_np[i]]['errors'].append(err ** 2)
                    snr_results[snr_np[i]]['abs_errors'].append(np.abs(err))
    # 打印表格
    print(f"{'SNR (dB)':<10} {'RMSE (deg)':<12} {'MAE (deg)':<12} {'Samples':<10}")
    print("-" * 50)
    snr_list = sorted(snr_results.keys())
    rmse_list, mae_list = [], []
    for snr in snr_list:
        rmse = np.sqrt(np.mean(snr_results[snr]['errors']))
        mae = np.mean(snr_results[snr]['abs_errors'])
        n = len(snr_results[snr]['errors'])
        print(f"{snr:<10} {rmse:<12.4f} {mae:<12.4f} {n:<10}")
        rmse_list.append(rmse)
        mae_list.append(mae)
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(snr_list, rmse_list, 'bo-', label='RMSE')
    plt.plot(snr_list, mae_list, 'rs-', label='MAE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Angle Error (deg)')
    plt.title('SNR-wise DOA Estimation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_dir, 'snr_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ---------- 绘制训练曲线 ----------
def plot_training_curves_doa(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 只绘制验证集 RMSE（如果存在），训练集 RMSE 可省略
    if 'val_rmse' in history and len(history['val_rmse']) > 0:
        axes[1].plot(epochs, history['val_rmse'], label='Val RMSE', color='r')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE (degree)')
        axes[1].set_title('DOA RMSE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # 如果没有 RMSE 数据，可以隐藏第二个子图或放其他信息
        axes[1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


# 数据集解包/可依自己数据集的生成习惯修改/
# class H5DatasetMultiSNR(Dataset):
#     def __init__(self, folder_path):
#         self.RX_list = []
#         self.XX_list = []
#         self.S_label_list = []
#         self.S_MR_label_list = []
#         self.snr_label_list = []

#         # 遍历 SNR 子文件夹
#         for snr_dir in os.listdir(folder_path):
#             if not snr_dir.startswith('SNR_'):
#                 continue

#             snr_value = int(snr_dir.split('_')[-1])
#             mat_file = os.path.join(folder_path, snr_dir, f'data_{snr_dir}.mat')

#             with h5py.File(mat_file, 'r') as f:
#                 RX_raw = np.array(f['RX_local'])
#                 XX_raw = np.array(f['XX_local'])
#                 RX = RX_raw['real'] + 1j * RX_raw['imag']
#                 XX = XX_raw['real'] + 1j * XX_raw['imag']

#                 RX = RX.transpose(2, 0, 1)
#                 XX = XX.transpose(2, 1, 0)
#                 S_label = np.array(f['S_label_local']).transpose()
#                 S_MR_label = np.array(f['MR_label_local']).transpose()

#             self.RX_list.append(RX.astype(np.complex64))
#             self.XX_list.append(XX.astype(np.complex64))
#             self.S_label_list.append(S_label.astype(np.float32))
#             self.S_MR_label_list.append(S_MR_label.squeeze().astype(np.int64))
#             self.snr_label_list.append(np.full(len(S_label), snr_value, dtype=np.int32))

#         # 合并所有数据
#         self.RX = np.concatenate(self.RX_list, axis=0)
#         self.XX = np.concatenate(self.XX_list, axis=0)
#         self.S_label = np.concatenate(self.S_label_list, axis=0)
#         self.S_MR_label = np.concatenate(self.S_MR_label_list, axis=0)
#         self.SNR_label = np.concatenate(self.snr_label_list, axis=0)

#     def __len__(self):
#         return len(self.RX)

#     def __getitem__(self, idx):
#         rx = self.RX[idx]
#         xx = self.XX[idx]

#         rx_data = np.stack([rx.real, rx.imag], axis=0)
#         xx_data = np.stack([xx.real, xx.imag], axis=0)

#         return (
#             torch.tensor(rx_data, dtype=torch.float32),
#             torch.tensor(xx_data, dtype=torch.float32),
#             torch.tensor(self.S_MR_label[idx], dtype=torch.long),
#             torch.tensor(self.S_label[idx], dtype=torch.float32),
#             torch.tensor(self.SNR_label[idx], dtype=torch.int32)
            
#         )


# def plot_history(history, task, save_dir):
#     """
#     绘制训练历史曲线，支持：
#     """
#     # 创建保存目录（确保存在）
#     os.makedirs(save_dir, exist_ok=True)

#     # ----------------------- 1. MOD任务（分类：损失+准确率） -----------------------
#     if task in ['mod', 'both']:
#         # 绘制MOD交叉熵损失曲线
#         if 'mod' in history['train']['loss']:
#             plt.figure(figsize=(8, 5))
#             plt.plot(history['train']['loss']['mod'], label='Train Loss (CE)')
#             plt.plot(history['val']['loss']['mod'], label='Val Loss (CE)')
#             plt.title("MOD Cross-Entropy Loss Curve")
#             plt.xlabel("Epoch")
#             plt.ylabel("Cross-Entropy Loss")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f'mod_loss_curve.png'))
#             plt.close()

#         # 绘制MOD准确率曲线
#         if 'mod' in history['train']['acc']:
#             plt.figure(figsize=(8, 5))
#             plt.plot(history['train']['acc']['mod'], label='Train Accuracy')
#             plt.plot(history['val']['acc']['mod'], label='Val Accuracy')
#             plt.title("MOD Accuracy Curve")
#             plt.xlabel("Epoch")
#             plt.ylabel("Accuracy")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f'mod_acc_curve.png'))
#             plt.close()

#     # ----------------------- 2. DOA任务（分类：损失+准确率+RMSE） -----------------------
#     if task in ['doa', 'both']:
#         # 绘制DOA交叉熵损失曲线
#         if 'doa' in history['train']['loss']:
#             plt.figure(figsize=(8, 5))
#             plt.plot(history['train']['loss']['doa'], label='Train Loss (CE)')
#             plt.plot(history['val']['loss']['doa'], label='Val Loss (CE)')
#             plt.title("DOA Cross-Entropy Loss Curve")
#             plt.xlabel("Epoch")
#             plt.ylabel("Cross-Entropy Loss")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f'doa_loss_curve.png'))
#             plt.close()

#         # 绘制DOA准确率曲线（分类任务核心指标）
#         if 'doa' in history['train']['acc']:
#             plt.figure(figsize=(8, 5))
#             plt.plot(history['train']['acc']['doa'], label='Train Accuracy')
#             plt.plot(history['val']['acc']['doa'], label='Val Accuracy')
#             plt.title("DOA Accuracy Curve")
#             plt.xlabel("Epoch")
#             plt.ylabel("Accuracy")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f'doa_acc_curve.png'))
#             plt.close()

#         # 绘制DOA RMSE曲线（角度误差辅助指标）
#         if 'doa' in history['train']['rmse']:
#             plt.figure(figsize=(8, 5))
#             plt.plot(history['train']['rmse']['doa'], label='Train RMSE')
#             plt.plot(history['val']['rmse']['doa'], label='Val RMSE')
#             plt.title("DOA Angle RMSE Curve")
#             plt.xlabel("Epoch")
#             plt.ylabel("RMSE (°)")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f'doa_rmse_curve.png'))
#             plt.close()

#     # ----------------------- 3. BOTH任务（总损失曲线） -----------------------
#     if task == 'both':
#         # 绘制总损失曲线（MOD+DOA加权和）
#         if 'total' in history['train']['loss']:
#             plt.figure(figsize=(8, 5))
#             plt.plot(history['train']['loss']['total'], label='Train Total Loss')
#             plt.plot(history['val']['loss']['total'], label='Val Total Loss')
#             plt.title("Combined Total Loss (MOD + DOA) Curve")
#             plt.xlabel("Epoch")
#             plt.ylabel("Total Loss")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(os.path.join(save_dir, f'both_total_loss_curve.png'))
#             plt.close()

#     print(f"History plots saved to {save_dir}")


# # 角度转索引
# def doa_angle_to_idx(angle_array):
#     return angle_array + 40

# def doa_idx_to_angle(idx_array):
#     return idx_array - 40  # idx ∈ [0, 80] → angle ∈ [-40, 40]

# # 角度转换部分有点绕注意逻辑对应
# def evaluate_by_snr(model, test_loader, device, task, result_dir):
#     model.eval()
#     os.makedirs(result_dir, exist_ok=True)

#     # ----------- 存储容器 ----------- #
#     # Modulation
#     mod_true, mod_pred, mod_snr = [], [], []

#     # DOA
#     true_doa_angle, pred_doa_angle, snr_labels = [], [], []

#     mod_labels = [
#         "FSK", "BPSK", "LFM", "FRANK",
#         "P1", "P2", "P3", "P4",
#         "T1", "T2", "T3", "T4"
#     ]

#     with torch.no_grad():
#         for sample in test_loader:
#             doa_input = sample[0].to(device)
#             mod_input = sample[1].to(device)
#             true_mod_batch = sample[2].cpu().numpy()
#             true_doa_batch = sample[3].cpu().numpy()
#             snr_batch = sample[4].cpu().numpy()

#             mod_out, doa_out, result_mod, result_doa = model(
#                 mod_input=mod_input,
#                 doa_input=doa_input,
#                 task=task
#             )

#             # -------- Modulation -------- #
#             if task in ['mod', 'both']:
#                 mod_pred_batch = result_mod.cpu().numpy()
#                 mod_true.extend(true_mod_batch)
#                 mod_pred.extend(mod_pred_batch)
#                 mod_snr.extend(snr_batch)

#             # -------- DOA -------- #
#             if task in ['doa', 'both']:
#                 pred_doa_batch = result_doa.cpu().numpy()

#                 # 转换角度（idx <-> angle）
#                 if np.any(true_doa_batch < 0):
#                     true_doa_idx = doa_angle_to_idx(true_doa_batch)
#                 else:
#                     true_doa_idx = true_doa_batch
#                 pred_doa_idx = pred_doa_batch

#                 true_angle_batch = doa_idx_to_angle(true_doa_idx)
#                 pred_angle_batch = doa_idx_to_angle(pred_doa_idx)

#                 true_doa_angle.extend(true_angle_batch)
#                 pred_doa_angle.extend(pred_angle_batch)
#                 snr_labels.extend(snr_batch)

#     # ------------------- 根据任务选择 unique_snr ------------------- #
#     if task == 'mod':
#         unique_snr = np.sort(np.unique(mod_snr)) if len(mod_snr) > 0 else []
#     elif task == 'doa':
#         unique_snr = np.sort(np.unique(snr_labels)) if len(snr_labels) > 0 else []
#     elif task == 'both':
#         # 两个都有的话，用 DOA 的 snr_labels（和 mod_snr 应该一致）
#         unique_snr = np.sort(np.unique(snr_labels)) if len(snr_labels) > 0 else []
#     else:
#         unique_snr = []

#     # ------------------- Modulation 评估 ------------------- #
#     if task in ['mod', 'both'] and len(mod_true) > 0:
#         mod_true = np.array(mod_true)
#         mod_pred = np.array(mod_pred)
#         mod_snr = np.array(mod_snr)

#         # 0 dB 混淆矩阵
#         zero_db_mask = (mod_snr == 0)
#         if np.sum(zero_db_mask) > 0:
#             cm = confusion_matrix(mod_true[zero_db_mask], mod_pred[zero_db_mask], labels=np.arange(12))
#             cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
#             plt.figure(figsize=(12, 10))
#             sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
#                         xticklabels=mod_labels, yticklabels=mod_labels,
#                         cbar_kws={'label': 'Percentage (%)'})
#             plt.xlabel('Predicted Modulation')
#             plt.ylabel('True Modulation')
#             plt.title('Confusion Matrix (0 dB SNR)')
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plt.savefig(os.path.join(result_dir, 'mod_confusion_matrix_0db.png'), dpi=300)
#             plt.close()
#             print("0dB SNR modulation confusion matrix saved.")

#         # Accuracy vs SNR
#         mod_acc_per_snr = []
#         for snr in unique_snr:
#             mask = (mod_snr == snr)
#             if np.sum(mask) == 0:
#                 continue
#             acc = np.mean(mod_pred[mask] == mod_true[mask])
#             mod_acc_per_snr.append(acc)

#         plt.figure(figsize=(10, 5))
#         plt.plot(unique_snr, mod_acc_per_snr, marker='^', color='g', label='Modulation Accuracy')
#         plt.xlabel('SNR (dB)')
#         plt.ylabel('Accuracy')
#         plt.title('Modulation Accuracy vs SNR')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(os.path.join(result_dir, 'modulation_accuracy_vs_snr.png'))
#         plt.close()

#         # 每类调制 Accuracy vs SNR
#         num_classes = len(mod_labels)
#         class_acc_dict = {label: [] for label in mod_labels}
#         for snr in unique_snr:
#             mask = (mod_snr == snr)
#             for i, label in enumerate(mod_labels):
#                 class_mask = mask & (mod_true == i)
#                 acc = np.mean(mod_pred[class_mask] == mod_true[class_mask]) if np.sum(class_mask) > 0 else np.nan
#                 class_acc_dict[label].append(acc)

#         plt.figure(figsize=(14, 8))
#         colors = plt.cm.get_cmap('tab20', num_classes)
#         for i, label in enumerate(mod_labels):
#             plt.plot(unique_snr, class_acc_dict[label], marker='o', label=label, color=colors(i))
#         plt.xlabel('SNR (dB)')
#         plt.ylabel('Accuracy')
#         plt.title('Modulation Accuracy per Class vs SNR')
#         plt.grid(True)
#         plt.legend(loc='lower right')
#         plt.savefig(os.path.join(result_dir, 'modulation_accuracy_per_class_vs_snr.png'))
#         plt.close()

#         # 保存结果到 txt
#         with open(os.path.join(result_dir, 'modulation_accuracy.txt'), 'w') as f:
#             f.write("SNR(dB)\tModulation_Accuracy\n")
#             for snr, acc in zip(unique_snr, mod_acc_per_snr):
#                 f.write(f"{snr}\t{acc:.4f}\n")

#         with open(os.path.join(result_dir, 'modulation_accuracy_per_class.txt'), 'w') as f:
#             header = "SNR(dB)\t" + "\t".join(mod_labels) + "\n"
#             f.write(header)
#             for i, snr in enumerate(unique_snr):
#                 accs = [class_acc_dict[label][i] if not np.isnan(class_acc_dict[label][i]) else -1 for label in mod_labels]
#                 accs_str = "\t".join(f"{acc:.4f}" if acc >= 0 else "NaN" for acc in accs)
#                 f.write(f"{snr}\t{accs_str}\n")

#     # ------------------- DOA 评估 ------------------- #
#     if task in ['doa', 'both'] and len(unique_snr) > 0:
#         true_doa_angle = np.array(true_doa_angle)
#         pred_doa_angle = np.array(pred_doa_angle)
#         snr_labels = np.array(snr_labels)

#         doa_rmse_per_snr = []
#         for snr in unique_snr:
#             mask = (snr_labels == snr)
#             if np.sum(mask) == 0:
#                 continue
#             rmse = np.sqrt(mean_squared_error(true_doa_angle[mask], pred_doa_angle[mask]))
#             doa_rmse_per_snr.append(rmse)
#             print(f"SNR={snr} dB, DOA_RMSE={rmse:.4f}")

#         plt.figure(figsize=(10, 5))
#         plt.plot(unique_snr, doa_rmse_per_snr, marker='s', color='r', label='DOA RMSE')
#         plt.xlabel('SNR (dB)')
#         plt.ylabel('RMSE (°)')
#         plt.title('DOA Angle RMSE vs SNR')
#         plt.grid(True)
#         plt.legend()
#         plt.ylim(bottom=0)
#         plt.savefig(os.path.join(result_dir, 'doa_rmse_vs_snr.png'))
#         plt.close()

#         # 保存 txt
#         with open(os.path.join(result_dir, 'doa_rmse.txt'), 'w') as f:
#             f.write("SNR(dB)\tDOA_RMSE\n")
#             for snr, rmse in zip(unique_snr, doa_rmse_per_snr):
#                 f.write(f"{snr}\t{rmse:.4f}\n")

#     print(f"SNR-based evaluation completed. Results saved to {result_dir}")

