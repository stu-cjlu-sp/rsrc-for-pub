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
from tool import * 


#=======波束估计部分使用的工具 ========
# 从谱里面找真实的角度值
def findpeaks(x, K, DOA):
    x = np.array(x)
    n = len(x)
    
    # 自定义峰值检测参数，优化边界和特殊点的检测
    # height=None: 不限制峰值高度
    # threshold=None: 不限制峰与谷的差值
    # distance=1: 确保峰之间至少间隔1个点
    # prominence=None: 不限制峰的显著度
    # width=None: 不限制峰的宽度
    # wlen=None: 不指定评估窗口长度
    indexes, _ = scipy.signal.find_peaks(
        x, 
        distance=1  # 确保峰之间有足够间隔
    )
    
    # 处理边界情况：检查首尾是否为最大值（即使不被识别为峰值）
    # 首端检查：如果第一个元素比右边邻居大，视为峰值
    if n >= 2 and x[0] > x[1]:
        indexes = np.insert(indexes, 0, 0)  # 添加到峰值列表
    # 末端检查：如果最后一个元素比左边邻居大，视为峰值
    if n >= 2 and x[-1] > x[-2]:
        indexes = np.append(indexes, n-1)  # 添加到峰值列表
    
    # 去重（防止边界检查添加了已存在的峰值）
    indexes = np.unique(indexes)
    
    # 如果仍然没有找到峰值，直接取最大值位置
    if len(indexes) == 0:
        max_idx = np.argmax(x)
        indexes = np.array([max_idx])
    
    # 按峰值大小排序，取前K个
    peak_values = x[indexes]
    sorted_indices = np.argsort(peak_values)[::-1]  # 降序排列
    top_k_indexes = indexes[sorted_indices[:K]]
    
    p = np.zeros(K)
    ind = np.zeros(K, dtype=int)
    
    for i in range(K):
        idx = top_k_indexes[i]
        ind[i] = idx
        
        # 边界处理与插值
        if idx == 0:
            # 左边界，只能向右插值
            if n == 1:
                p[i] = DOA[idx]
            else:
                p[i] = (x[idx] * DOA[idx] + x[idx+1] * DOA[idx+1]) / (x[idx] + x[idx+1])
        elif idx == n - 1:
            # 右边界，只能向左插值
            p[i] = (x[idx-1] * DOA[idx-1] + x[idx] * DOA[idx]) / (x[idx-1] + x[idx])
        else:
            # 中间位置，根据左右邻居决定插值方向
            left_val = x[idx-1]
            right_val = x[idx+1]
            current_val = x[idx]
            
            if right_val > left_val:
                # 向右插值
                p[i] = (right_val * DOA[idx+1] + current_val * DOA[idx]) / (right_val + current_val)
            else:
                # 向左插值
                p[i] = (left_val * DOA[idx-1] + current_val * DOA[idx]) / (left_val + current_val)
    
    return p, DOA[ind]

#====onlydoa=====
#训练测试函数
# ---------- 训练一个 epoch ----------
def train_epochdoa(model, dataloader, optimizer, criterion, device):
    """
    训练一个 epoch，返回平均损失。
    不计算 RMSE，以保持训练效率。
    """
    model.train()
    total_loss = 0.0
    for cov, spectrum, _, _ in dataloader:
        cov = cov.to(device)
        spectrum = spectrum.to(device)
        pred = model(cov)
        loss = criterion(pred, spectrum)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ---------- 验证一个 epoch（返回 loss 和 RMSE）----------
def valid_epochdoa(model, dataloader, criterion, device, DOA_grid, num_sources=1):
    model.eval()
    total_loss = 0.0
    all_pred_angles = []
    all_true_angles = []

    with torch.no_grad():
        for cov, spectrum, true_angle, _ in dataloader:
            cov = cov.to(device)
            spectrum = spectrum.to(device)
            pred_spec = model(cov)
            loss = criterion(pred_spec, spectrum)
            total_loss += loss.item()
            pred_spec_np = pred_spec.cpu().numpy()
            for i in range(pred_spec_np.shape[0]):
                angles, _ = findpeaks(pred_spec_np[i], K=num_sources, DOA=DOA_grid)
                all_pred_angles.extend(angles[:num_sources])
            all_true_angles.extend(true_angle.cpu().numpy())
    pred_arr = np.array(all_pred_angles)
    true_arr = np.array(all_true_angles)
    rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))

    return total_loss / len(dataloader), rmse

# ---------- 测试 epoch（返回 loss, RMSE, MAE）----------
def test_epochdoa(model, dataloader, criterion, device, DOA_grid, num_sources=1):
    model.eval()
    total_loss = 0.0
    all_pred_angles = []
    all_true_angles = []
    with torch.no_grad():
        for cov, spectrum, true_angle, _ in dataloader:
            cov = cov.to(device)
            spectrum = spectrum.to(device)
            pred_spec = model(cov)
            loss = criterion(pred_spec, spectrum)
            total_loss += loss.item()
            pred_spec_np = pred_spec.cpu().numpy()
            for i in range(pred_spec_np.shape[0]):
                angles, _ = findpeaks(pred_spec_np[i], K=num_sources, DOA=DOA_grid)
                all_pred_angles.extend(angles[:num_sources])
            all_true_angles.extend(true_angle.cpu().numpy())
    pred_arr = np.array(all_pred_angles)
    true_arr = np.array(all_true_angles)
    rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
    mae = np.mean(np.abs(pred_arr - true_arr))
    return total_loss / len(dataloader), rmse, mae

# pahse1 only RX
def train_epoch_doa(model, loader, optimizer, criterion, device, DOA_grid):
    model.train()
    total_loss = 0
    total_rmse = 0
    for cov, spectrum, true_angle, _ in loader:
        cov = cov.to(device)
        spectrum = spectrum.to(device)
        pred = model(cov)
        loss = criterion(pred, spectrum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 计算 RMSE（使用峰值检测）
        pred_np = pred.detach().cpu().numpy()
        true_angle_np = true_angle.cpu().numpy()
        for i in range(pred_np.shape[0]):
            angles, _ = findpeaks(pred_np[i], K=1, DOA=DOA_grid)
            pred_angle = angles[0]
            rmse = (pred_angle - true_angle_np[i]) ** 2
            total_rmse += rmse
    return total_loss / len(loader), np.sqrt(total_rmse / len(loader.dataset))

# pahse1 only RX
def valid_epoch_doa(model, loader, criterion, device, DOA_grid):
    model.eval()
    total_loss = 0
    total_rmse = 0
    with torch.no_grad():
        for cov, spectrum, true_angle, _ in loader:
            cov = cov.to(device)
            spectrum = spectrum.to(device)
            pred = model(cov)
            loss = criterion(pred, spectrum)
            total_loss += loss.item()
            pred_np = pred.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(pred_np.shape[0]):
                angles, _ = findpeaks(pred_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                rmse = (pred_angle - true_angle_np[i]) ** 2
                total_rmse += rmse
    return total_loss / len(loader), np.sqrt(total_rmse / len(loader.dataset))

# pahse1 only RX
def test_epoch_doa(model, loader, criterion, device, DOA_grid):
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_mae = 0
    with torch.no_grad():
        for cov, spectrum, true_angle, _ in loader:
            cov = cov.to(device)
            spectrum = spectrum.to(device)
            pred = model(cov)
            loss = criterion(pred, spectrum)
            total_loss += loss.item()
            pred_np = pred.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(pred_np.shape[0]):
                angles, _ = findpeaks(pred_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                err = pred_angle - true_angle_np[i]
                total_rmse += err**2
                total_mae += abs(err)
    n = len(loader.dataset)
    return total_loss / len(loader), np.sqrt(total_rmse / n), total_mae / n

#====onlymod=====
def train_epoch_mod(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for iq, label, snr in loader:   # 解包顺序: iq, mod, snr
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
        for iq, label, snr in loader:   # 解包顺序: iq, mod, snr
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
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
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

# ----------------------------- 5. 仅训练 FC 的 epoch 函数 -----------------------------
# ----------------------------- 训练函数 (FC only) -----------------------------
def train_epoch_fc_only(model, loader, optimizer, loss_fc, device):
    model.train()
    total_loss = 0.0
    for cov, spectrum, angle, iq, mod, snr, fc_norm in loader:
        iq = iq.to(device)
        fc_norm = fc_norm.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        fc_hat = model(iq)
        loss = loss_fc(fc_hat, fc_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ----------------------------- 6. 验证/测试函数（计算 FC RMSE 和 DOA RMSE）-----------------------------
def eval_epoch_fc_only(model, loader, device, dataset, loss_fc,
                       DOA_grid, M, d, num_signal,
                       use_true_fc=False, verbose=True):
    model.eval()
    total_loss = 0.0
    all_fc_true_orig = []
    all_fc_pred_orig = []
    all_doa_true = []
    all_doa_pred = []

    # 控制打印：只打印第一个 batch 的前几个样本
    printed_batch = False

    with torch.no_grad():
        for batch_idx, (cov, spectrum, angle, iq, mod, snr, fc_norm) in enumerate(loader):
            iq = iq.to(device)
            fc_norm = fc_norm.to(device).float().view(-1, 1)
            fc_hat = model(iq)
            loss = loss_fc(fc_hat, fc_norm)
            total_loss += loss.item()

            # 反归一化
            fc_true_np = fc_norm.cpu().numpy().flatten()
            fc_pred_np = fc_hat.cpu().numpy().flatten()
            fc_true_orig = dataset.get_fc_denorm(fc_true_np)
            fc_pred_orig = dataset.get_fc_denorm(fc_pred_np)
            all_fc_true_orig.extend(fc_true_orig)
            all_fc_pred_orig.extend(fc_pred_orig)

            batch_cov = cov.cpu().numpy()
            for i in range(len(fc_pred_orig)):
                # 选择用于 DOA 的载频
                fc_doa = fc_true_orig[i] if use_true_fc else fc_pred_orig[i]

                R_complex = batch_cov[i, 0] + 1j * batch_cov[i, 1]
                try:
                    P = music_spectrum_numpy(R_complex, M=M, d=d, fc_hat=fc_doa,
                                             angle_grid=DOA_grid, num_signal=num_signal)
                    ang, _ = findpeaks(P, K=1, DOA=DOA_grid)
                    doa_est = ang[0]
                except Exception as e:
                    # 降级：取谱最大值
                    P = music_spectrum_numpy(R_complex, M=M, d=d, fc_hat=fc_doa,
                                             angle_grid=DOA_grid, num_signal=num_signal)
                    doa_est = DOA_grid[np.argmax(P)]

                true_angle = angle[i].item()
                all_doa_pred.append(doa_est)
                all_doa_true.append(true_angle)

                # 打印第一个 batch 的前 5 个样本的详细信息
                if not printed_batch and batch_idx == 0 and i < 5:
                    print(f"[DEBUG] Sample {i}: true_angle={true_angle:.1f}°, pred_angle={doa_est:.1f}°")
                    print(f"        fc_true={fc_true_orig[i]:.2e} Hz, fc_pred={fc_pred_orig[i]:.2e} Hz, used_fc={fc_doa:.2e} Hz")
                    # 可选：打印谱的最大值和对应角度
                    max_idx = np.argmax(P)
                    print(f"        Spectrum max at {DOA_grid[max_idx]:.1f}° (value {P[max_idx]:.4f})")
                    if i == 4:
                        printed_batch = True

    fc_rmse = np.sqrt(np.mean((np.array(all_fc_pred_orig) - np.array(all_fc_true_orig))**2))
    doa_rmse = np.sqrt(np.mean((np.array(all_doa_pred) - np.array(all_doa_true))**2))
    return total_loss / len(loader), fc_rmse, doa_rmse

# ----------------------------- 7. 绘图函数 -----------------------------
def plot_fc_only_history(history, save_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (norm)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history['val_fc_rmse'], label='Val FC RMSE (Hz)')
    plt.plot(history['val_doa_rmse'], label='Val DOA RMSE (deg)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fc_only_training_curves.png'), dpi=150)
    plt.close()

# only input x fc and mod joint
# def train_epoch_joint_fc_mod(model, loader, optimizer, loss_fc, loss_cls, device, lambda_fc=1.0):
#     model.train()
#     total_loss = 0.0
#     total_fc_loss = 0.0
#     total_cls_loss = 0.0
#     correct = 0
#     total = 0
#     for cov, spectrum, angle, iq, mod, snr, fc_norm in loader:
#         iq = iq.to(device)
#         mod = mod.to(device)
#         fc_norm = fc_norm.to(device).float().view(-1, 1)

#         optimizer.zero_grad()
#         fc_hat, cls_out = model(iq)
#         loss1 = loss_cls(cls_out, mod)
#         loss2 = loss_fc(fc_hat, fc_norm)
#         loss = loss1 + lambda_fc * loss2
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_fc_loss += loss2.item()
#         total_cls_loss += loss1.item()
#         pred = cls_out.argmax(dim=1)
#         correct += (pred == mod).sum().item()
#         total += mod.size(0)

#     return (total_loss / len(loader),
#             total_fc_loss / len(loader),
#             total_cls_loss / len(loader),
#             correct / total * 100)

# # ===================== 验证/测试函数 =====================
# def eval_epoch_joint_fc_mod(model, loader, device, dataset, loss_fc, loss_cls,
#                      DOA_grid, M, d, num_signal, lambda_fc=1.0, use_true_fc=False):
#     model.eval()
#     total_loss = 0.0
#     total_fc_loss = 0.0
#     total_cls_loss = 0.0
#     correct = 0
#     total = 0
#     all_fc_true_orig = []
#     all_fc_pred_orig = []
#     all_doa_true = []
#     all_doa_pred = []

#     printed_batch = False

#     with torch.no_grad():
#         for batch_idx, (cov, spectrum, angle, iq, mod, snr, fc_norm) in enumerate(loader):
#             iq = iq.to(device)
#             mod = mod.to(device)
#             fc_norm = fc_norm.to(device).float().view(-1, 1)

#             fc_hat, cls_out = model(iq)
#             loss1 = loss_cls(cls_out, mod)
#             loss2 = loss_fc(fc_hat, fc_norm)
#             loss = loss1 + lambda_fc * loss2

#             total_loss += loss.item()
#             total_fc_loss += loss2.item()
#             total_cls_loss += loss1.item()
#             pred = cls_out.argmax(dim=1)
#             correct += (pred == mod).sum().item()
#             total += mod.size(0)

#             # 载频反归一化
#             fc_true_np = fc_norm.cpu().numpy().flatten()
#             fc_pred_np = fc_hat.cpu().numpy().flatten()
#             fc_true_orig = dataset.get_fc_denorm(fc_true_np)
#             fc_pred_orig = dataset.get_fc_denorm(fc_pred_np)
#             all_fc_true_orig.extend(fc_true_orig)
#             all_fc_pred_orig.extend(fc_pred_orig)

#             # DOA 估计
#             batch_cov = cov.cpu().numpy()
#             for i in range(len(fc_pred_orig)):
#                 fc_doa = fc_true_orig[i] if use_true_fc else fc_pred_orig[i]
#                 R_complex = batch_cov[i, 0] + 1j * batch_cov[i, 1]
#                 try:
#                     P = music_spectrum_numpy(R_complex, M=M, d=d, fc_hat=fc_doa,
#                                              angle_grid=DOA_grid, num_signal=num_signal)
#                     ang, _ = findpeaks(P, K=1, DOA=DOA_grid)
#                     doa_est = ang[0]
#                 except:
#                     P = music_spectrum_numpy(R_complex, M=M, d=d, fc_hat=fc_doa,
#                                              angle_grid=DOA_grid, num_signal=num_signal)
#                     doa_est = DOA_grid[np.argmax(P)]
#                 all_doa_pred.append(doa_est)
#                 all_doa_true.append(angle[i].item())

#                 # 调试打印（仅第一个 batch 的前5个样本）
#                 if not printed_batch and batch_idx == 0 and i < 5:
#                     print(f"[DEBUG] Sample {i}: true_angle={angle[i].item():.1f}°, pred_angle={doa_est:.1f}°")
#                     print(f"        fc_true={fc_true_orig[i]:.2e} Hz, fc_pred={fc_pred_orig[i]:.2e} Hz, used_fc={fc_doa:.2e} Hz")
#                     max_idx = np.argmax(P)
#                     print(f"        Spectrum max at {DOA_grid[max_idx]:.1f}° (value {P[max_idx]:.4f})")
#                     if i == 4:
#                         printed_batch = True

#     mod_acc = correct / total * 100
#     fc_rmse = np.sqrt(np.mean((np.array(all_fc_pred_orig) - np.array(all_fc_true_orig))**2))
#     doa_rmse = np.sqrt(np.mean((np.array(all_doa_pred) - np.array(all_doa_true))**2))
#     return (total_loss / len(loader),
#             total_fc_loss / len(loader),
#             total_cls_loss / len(loader),
#             mod_acc,
#             fc_rmse,
#             doa_rmse)

# # ===================== 绘图函数 =====================
# def plot_joint_fc_mod_history(history, save_dir):
#     plt.figure(figsize=(15, 10))

#     # 子图1：损失曲线
#     plt.subplot(2, 2, 1)
#     plt.plot(history['train_loss'], label='Train Total Loss')
#     plt.plot(history['val_loss'], label='Val Total Loss')
#     plt.plot(history['train_fc_loss'], '--', label='Train FC Loss')
#     plt.plot(history['val_fc_loss'], '--', label='Val FC Loss')
#     plt.plot(history['train_cls_loss'], ':', label='Train Cls Loss')
#     plt.plot(history['val_cls_loss'], ':', label='Val Cls Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # 子图2：调制准确率
#     plt.subplot(2, 2, 2)
#     plt.plot(history['train_mod_acc'], label='Train Mod Acc')
#     plt.plot(history['val_mod_acc'], label='Val Mod Acc')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True)

#     # 子图3：载频 RMSE（单位 Hz，转换为 MHz）
#     plt.subplot(2, 2, 3)
#     val_fc_rmse_mhz = [v / 1e6 for v in history['val_fc_rmse']]
#     plt.plot(val_fc_rmse_mhz, label='Val FC RMSE (MHz)')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMSE (MHz)')
#     plt.legend()
#     plt.grid(True)

#     # 子图4：DOA RMSE
#     plt.subplot(2, 2, 4)
#     plt.plot(history['val_doa_rmse'], label='Val DOA RMSE (deg)')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMSE (deg)')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'joint_training_curves.png'), dpi=150)
#     plt.close()
#     print(f"Training curves saved to {save_dir}/joint_training_curves.png")
    
# 二阶段训练
def cross_entropy_spectrum_loss(P_hat, P_true, temperature=1.0, eps=1e-8):
    """
    将预测谱和真实谱视为概率分布，计算 KL 散度（等价于交叉熵，因为真实谱熵固定）
    Args:
        P_hat: (B, A) 预测谱（正值，未归一化）
        P_true: (B, A) 真实谱（正值，未归一化）
        temperature: 温度参数，越低分布越尖锐，对峰值敏感
        eps: 防止除零
    Returns:
        loss: 标量
    """
    # 应用温度缩放
    P_hat = P_hat / temperature
    P_true = P_true / temperature
    # 稳定化处理
    P_hat = torch.softmax(P_hat, dim=1)
    P_true = torch.softmax(P_true, dim=1)
    # KL 散度
    loss = F.kl_div(P_hat.log(), P_true, reduction='batchmean')
    return loss

def train_epoch_finetune_stable(model, loader, optimizer, loss_fc, loss_cls, music_layer,
                                device, dataset, lambda_fc=1.0, epoch=0,
                                warmup_epochs=10, lambda_spec_max=0.05, temperature=0.5):
    """
    使用交叉熵谱损失进行微调训练
    """
    model.train()
    total_loss = 0.0
    total_fc_loss = 0.0
    total_cls_loss = 0.0
    total_spec_loss = 0.0
    correct = 0
    total = 0

    # 谱损失退火（线性增加）
    if epoch < warmup_epochs:
        lambda_spec = 0.0
    else:
        lambda_spec = lambda_spec_max * min(1.0, (epoch - warmup_epochs) / 20)

    for cov, spectrum, angle, iq, mod, snr, fc_norm in loader:
        iq = iq.to(device)
        cov = cov.to(device)
        mod = mod.to(device)
        fc_norm = fc_norm.to(device).float().view(-1, 1)
        spectrum = spectrum.to(device)

        optimizer.zero_grad()

        fc_hat_norm, cls_out = model(iq)
        fc_hat_real = dataset.denorm_tensor(fc_hat_norm.squeeze())

        R_real = cov[:, 0, :, :]
        R_imag = cov[:, 1, :, :]
        R_complex = torch.complex(R_real, R_imag)

        # 可微 MUSIC 生成预测谱
        P_hat = music_layer(R_complex, fc_hat_real)   # (B, num_angles)

        loss1 = loss_cls(cls_out, mod)
        loss2 = loss_fc(fc_hat_norm, fc_norm)
        loss3 = cross_entropy_spectrum_loss(P_hat, spectrum, temperature=temperature)

        loss = loss1 + lambda_fc * loss2 + lambda_spec * loss3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_fc_loss += loss2.item()
        total_cls_loss += loss1.item()
        total_spec_loss += loss3.item()

        pred = cls_out.argmax(dim=1)
        correct += (pred == mod).sum().item()
        total += mod.size(0)

    mod_acc = correct / total * 100
    return (total_loss / len(loader),
            total_fc_loss / len(loader),
            total_cls_loss / len(loader),
            total_spec_loss / len(loader),
            mod_acc)


def eval_epoch_finetune_stable(model, loader, device, dataset, loss_fc, loss_cls, music_layer,
                               DOA_grid, M, d, num_signal, lambda_fc=1.0, use_true_fc=False,
                               epoch=0, warmup_epochs=10, lambda_spec_max=0.05, temperature=0.5):
    """
    使用交叉熵谱损失进行验证/测试（不更新梯度）
    """
    model.eval()
    total_loss = 0.0
    total_fc_loss = 0.0
    total_cls_loss = 0.0
    total_spec_loss = 0.0
    correct = 0
    total = 0
    all_fc_true_orig = []
    all_fc_pred_orig = []
    all_doa_true = []
    all_doa_pred = []

    # 验证时使用与训练相同的退火权重（仅用于记录）
    if epoch < warmup_epochs:
        lambda_spec = 0.0
    else:
        lambda_spec = lambda_spec_max * min(1.0, (epoch - warmup_epochs) / 20)

    with torch.no_grad():
        for cov, spectrum, angle, iq, mod, snr, fc_norm in loader:
            iq = iq.to(device)
            cov = cov.to(device)
            mod = mod.to(device)
            fc_norm = fc_norm.to(device).float().view(-1, 1)
            spectrum = spectrum.to(device)

            fc_hat_norm, cls_out = model(iq)
            fc_hat_real = dataset.denorm_tensor(fc_hat_norm.squeeze())

            R_real = cov[:, 0, :, :]
            R_imag = cov[:, 1, :, :]
            R_complex = torch.complex(R_real, R_imag)
            P_hat = music_layer(R_complex, fc_hat_real)

            loss1 = loss_cls(cls_out, mod)
            loss2 = loss_fc(fc_hat_norm, fc_norm)
            loss3 = cross_entropy_spectrum_loss(P_hat, spectrum, temperature=temperature)

            loss = loss1 + lambda_fc * loss2 + lambda_spec * loss3

            total_loss += loss.item()
            total_fc_loss += loss2.item()
            total_cls_loss += loss1.item()
            total_spec_loss += loss3.item()

            pred = cls_out.argmax(dim=1)
            correct += (pred == mod).sum().item()
            total += mod.size(0)

            # 载频反归一化（用于计算 RMSE）
            fc_true_np = fc_norm.cpu().numpy().flatten()
            fc_pred_np = fc_hat_norm.cpu().numpy().flatten()
            fc_true_orig = dataset.get_fc_denorm(fc_true_np)
            fc_pred_orig = dataset.get_fc_denorm(fc_pred_np)
            all_fc_true_orig.extend(fc_true_orig)
            all_fc_pred_orig.extend(fc_pred_orig)

            # DOA 估计（后处理，使用预测载频或真实载频）
            batch_cov = cov.cpu().numpy()
            for i in range(len(fc_pred_orig)):
                fc_doa = fc_true_orig[i] if use_true_fc else fc_pred_orig[i]
                R_complex_np = batch_cov[i, 0] + 1j * batch_cov[i, 1]
                try:
                    P = music_spectrum_numpy(R_complex_np, M=M, d=d, fc_hat=fc_doa,
                                             angle_grid=DOA_grid, num_signal=num_signal)
                    ang, _ = findpeaks(P, K=1, DOA=DOA_grid)
                    doa_est = ang[0]
                except Exception:
                    P = music_spectrum_numpy(R_complex_np, M=M, d=d, fc_hat=fc_doa,
                                             angle_grid=DOA_grid, num_signal=num_signal)
                    doa_est = DOA_grid[np.argmax(P)]
                all_doa_pred.append(doa_est)
                all_doa_true.append(angle[i].item())

    mod_acc = correct / total * 100
    fc_rmse = np.sqrt(np.mean((np.array(all_fc_pred_orig) - np.array(all_fc_true_orig))**2))
    doa_rmse = np.sqrt(np.mean((np.array(all_doa_pred) - np.array(all_doa_true))**2))
    return (total_loss / len(loader),
            total_fc_loss / len(loader),
            total_cls_loss / len(loader),
            total_spec_loss / len(loader),
            mod_acc,
            fc_rmse,
            doa_rmse)


def plot_finetune_history(history, save_dir):
    plt.figure(figsize=(18, 12))
    # 子图1：总损失
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Total')
    plt.plot(history['val_loss'], label='Val Total')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Total Loss')
    plt.legend(); plt.grid(True)
    # 子图2：FC/CLS 损失
    plt.subplot(2, 3, 2)
    plt.plot(history['train_fc_loss'], '--', label='Train FC')
    plt.plot(history['val_fc_loss'], '--', label='Val FC')
    plt.plot(history['train_cls_loss'], ':', label='Train CLS')
    plt.plot(history['val_cls_loss'], ':', label='Val CLS')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('FC & CLS Loss')
    plt.legend(); plt.grid(True)
    # 子图3：谱损失
    plt.subplot(2, 3, 3)
    plt.plot(history['train_spec_loss'], label='Train Spec')
    plt.plot(history['val_spec_loss'], label='Val Spec')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Spectral Loss')
    plt.legend(); plt.grid(True)
    # 子图4：调制准确率
    plt.subplot(2, 3, 4)
    plt.plot(history['train_mod_acc'], label='Train Acc')
    plt.plot(history['val_mod_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Modulation Accuracy')
    plt.legend(); plt.grid(True)
    # 子图5：FC RMSE
    plt.subplot(2, 3, 5)
    val_fc_rmse_mhz = [v/1e6 for v in history['val_fc_rmse']]
    plt.plot(val_fc_rmse_mhz, label='Val FC RMSE (MHz)')
    plt.xlabel('Epoch'); plt.ylabel('RMSE (MHz)'); plt.title('Carrier Frequency RMSE')
    plt.legend(); plt.grid(True)
    # 子图6：DOA RMSE
    plt.subplot(2, 3, 6)
    plt.plot(history['val_doa_rmse'], label='Val DOA RMSE')
    plt.xlabel('Epoch'); plt.ylabel('RMSE (deg)'); plt.title('DOA Performance')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_curves.png'), dpi=150)
    plt.close()

#最后的联合模型 MOD DOA FC融合拼接 用的是FC这个真实的标签
# ------------------------- 4. 训练、验证、测试函数 -------------------------
def train_epoch_jointnet(model, loader, optimizer, loss_cls, loss_doa, device, mode='joint'):
    """
    mode: 'doa', 'mod', 'joint'
    """
    model.train()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    pbar = tqdm(loader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        _ = batch[2]
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)
        _ = batch[5]
        fc_label = batch[6].to(device).float()

        mod_out, doa_out = model(iq, cov, fc_label)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred = torch.argmax(mod_out, dim=1)
            correct = (pred == mod_label).sum().item()
            total_correct += correct
            pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc

@torch.no_grad()
def eval_epoch_jointnet(model, loader, loss_cls, loss_doa, device, DOA_grid, mode='joint'):
    model.eval()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    total_doa_rmse = 0.0
    total_mae_doa = 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)
        _ = batch[5]
        fc_label = batch[6].to(device).float()

        mod_out, doa_out = model(iq, cov, fc_label)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += loss_d

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(batch_size):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    rmse_doa = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    mae_doa = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc, rmse_doa, mae_doa

@torch.no_grad()
def test_jointnet(model, loader, device, DOA_grid, mode='joint'):
    model.eval()
    total_correct, total_samples = 0, 0
    total_doa_rmse = 0.0
    total_mae_doa = 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)
        _= batch[5]
        fc_label = batch[6].to(device).float()

        mod_out, doa_out = model(iq, cov, fc_label)

        if mode in ['mod', 'joint']:
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(iq.size(0)):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

        total_samples += iq.size(0)

    mod_acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    doa_rmse = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    doa_mae = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return mod_acc, doa_rmse, doa_mae

# ------------------------- 5. 绘图函数 -------------------------
def plot_jointnet_history(history, save_dir, mode='joint'):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(14, 10))

    # 子图1：总损失（所有模式都有）
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss')

    # 子图2：调制损失或 DOA 损失（根据模式选择显示）
    plt.subplot(2, 2, 2)
    if mode in ['mod', 'joint'] and 'train_cls_loss' in history:
        plt.plot(epochs, history['train_cls_loss'], label='Train Cls Loss')
        plt.plot(epochs, history['val_cls_loss'], label='Val Cls Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Modulation Loss')
    elif mode in ['doa', 'joint'] and 'train_doa_loss' in history:
        plt.plot(epochs, history['train_doa_loss'], label='Train DOA Loss')
        plt.plot(epochs, history['val_doa_loss'], label='Val DOA Loss')
        plt.xlabel('Epoch')
        plt.ylabel('DOA Loss')
        plt.legend()
        plt.title('DOA Spectrum Loss')
    else:
        plt.text(0.5, 0.5, 'Not available in this mode', ha='center', va='center')
        plt.title('Loss (N/A)')

    # 子图3：另一个损失的显示（如果子图2显示了分类，这里显示DOA；反之亦然）
    plt.subplot(2, 2, 3)
    if mode in ['mod', 'joint'] and 'train_doa_loss' in history:
        plt.plot(epochs, history['train_doa_loss'], label='Train DOA Loss')
        plt.plot(epochs, history['val_doa_loss'], label='Val DOA Loss')
        plt.xlabel('Epoch')
        plt.ylabel('DOA Loss')
        plt.legend()
        plt.title('DOA Spectrum Loss')
    elif mode in ['doa', 'joint'] and 'train_cls_loss' in history:
        plt.plot(epochs, history['train_cls_loss'], label='Train Cls Loss')
        plt.plot(epochs, history['val_cls_loss'], label='Val Cls Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Modulation Loss')
    else:
        plt.text(0.5, 0.5, 'Not available in this mode', ha='center', va='center')
        plt.title('Loss (N/A)')

    # 子图4：性能指标（准确率 / DOA RMSE）
    plt.subplot(2, 2, 4)
    if mode in ['mod', 'joint'] and 'train_mod_acc' in history:
        plt.plot(epochs, history['train_mod_acc'], label='Train Acc')
        plt.plot(epochs, history['val_mod_acc'], label='Val Acc')
    if mode in ['doa', 'joint'] and 'val_doa_rmse' in history:
        plt.plot(epochs, history['val_doa_rmse'], label='Val DOA RMSE (deg)')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Performance Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")

#最后的联合模型 MOD DOA FC融合拼接 用的是FC这个预测的标签
def train_epoch_jointnet_pred(model, loader, optimizer, loss_cls, loss_doa, loss_fc, device,
                              mode='joint', lambda_cls=1.0, lambda_doa=1.0, lambda_fc=1.0):
    """
    训练一个 epoch。
    mode: 'mod', 'doa', 'joint'
    lambda_cls, lambda_doa, lambda_fc: 各损失的权重系数（仅在对应任务激活时使用）
    """
    model.train()
    total_loss, total_cls_loss, total_doa_loss, total_fc_loss = 0.0, 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    pbar = tqdm(loader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        _ = batch[2]
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)
        _ = batch[5]
        fc_label = batch[6].to(device).float()

        mod_out, doa_out, fc_hat = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d
        if mode == 'joint':
            loss_f = loss_fc(fc_hat, fc_label)
            total_fc_loss += loss_f.item() * iq.size(0)
            loss += lambda_fc * loss_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred = torch.argmax(mod_out, dim=1)
            correct = (pred == mod_label).sum().item()
            total_correct += correct
            pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    avg_fc  = total_fc_loss / total_samples if mode == 'joint' else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, avg_fc, acc


@torch.no_grad()
def eval_epoch_jointnet_pred(model, loader, loss_cls, loss_doa, loss_fc, device, DOA_grid,
                             mode='joint', lambda_cls=1.0, lambda_doa=1.0, lambda_fc=1.0):
    """
    验证一个 epoch，返回损失和指标。
    """
    model.eval()
    total_loss, total_cls_loss, total_doa_loss, total_fc_loss = 0.0, 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    total_doa_rmse = 0.0
    total_mae_doa = 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)
        _ = batch[5]
        fc_label = batch[6].to(device).float()

        mod_out, doa_out, fc_hat = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d
        if mode == 'joint':
            loss_f = loss_fc(fc_hat, fc_label)
            total_fc_loss += loss_f.item() * iq.size(0)
            loss += lambda_fc * loss_f

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(batch_size):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    avg_fc  = total_fc_loss / total_samples if mode == 'joint' else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    rmse_doa = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    mae_doa = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, avg_fc, acc, rmse_doa, mae_doa


@torch.no_grad()
def test_jointnet_pred(model, loader, device, DOA_grid, mode='joint', dataset=None):
    """
    测试函数，返回总体调制准确率、DOA RMSE 和 DOA MAE。
    不涉及损失权重，只输出最终性能指标。
    """
    model.eval()
    total_correct, total_samples = 0, 0
    total_doa_rmse = 0.0
    total_mae_doa = 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        if mode in ['mod', 'joint']:
            mod_out, doa_out, _ = model(iq, cov)
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct
        else:
            _, doa_out, _ = model(iq, cov)

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(iq.size(0)):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

        total_samples += iq.size(0)

    mod_acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    doa_rmse = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    doa_mae = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return mod_acc, doa_rmse, doa_mae

# ------------------------- 5. 按 SNR 分组的测试函数 -------------------------
def test_by_snr(model, loader, device, DOA_grid, mode='joint', dataset=None):
    """
    返回两个字典：snr_to_mod_acc, snr_to_doa_rmse
    """
    model.eval()
    snr_to_mod = defaultdict(lambda: {'correct': 0, 'total': 0})
    snr_to_doa = defaultdict(lambda: {'sq_err': 0.0, 'total': 0})

    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    with torch.no_grad():
        for batch in loader:
            cov = batch[0].to(device)
            angle_true = batch[2].to(device)
            iq = batch[3].to(device)
            mod_label = batch[4].to(device)
            snr_batch = batch[5].to(device) if len(batch) > 5 else None   # 需要 dataset 返回 snr
            # 注意：原 H5MultiTaskDataset 在 return_snr=True 时会返回 snr，位置在 mod_label 之后
            # 这里假设 batch 顺序为 cov, spectrum, angle, iq, mod, snr, fc
            # 根据前面定义，实际顺序是：cov, spectrum, angle, iq, mod, fc（如果没有 snr）
            # 要使用 snr 分组，需要修改 dataset 令 return_snr=True，并调整 batch 解包。
            # 为简化，我们先假设测试时不按 SNR 分组，仅返回总体指标，用户需要时可扩展。

    # 若需要完整实现，请确保 dataset 返回 snr 字段，并在 main 中传入带 snr 的 Subset。
    print("Warning: test_by_snr requires dataset to return snr. Skipping detailed SNR stats.")
    return {}, {}

# ------------------------- 6. 绘图函数 -------------------------
def plot_jointnet_pred_history(history, save_dir, mode='joint'):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(14, 12))

    # 总损失
    plt.subplot(3, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss')

    # 分类损失
    plt.subplot(3, 2, 2)
    if mode in ['mod', 'joint'] and 'train_cls_loss' in history:
        plt.plot(epochs, history['train_cls_loss'], label='Train Cls')
        plt.plot(epochs, history['val_cls_loss'], label='Val Cls')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Modulation Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # DOA 损失
    plt.subplot(3, 2, 3)
    if mode in ['doa', 'joint'] and 'train_doa_loss' in history:
        plt.plot(epochs, history['train_doa_loss'], label='Train DOA')
        plt.plot(epochs, history['val_doa_loss'], label='Val DOA')
        plt.xlabel('Epoch')
        plt.ylabel('DOA Loss')
        plt.legend()
        plt.title('DOA Spectrum Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # 载频回归损失（仅 joint 模式）
    plt.subplot(3, 2, 4)
    if mode == 'joint' and 'train_fc_loss' in history:
        plt.plot(epochs, history['train_fc_loss'], label='Train FC')
        plt.plot(epochs, history['val_fc_loss'], label='Val FC')
        plt.xlabel('Epoch')
        plt.ylabel('FC Loss')
        plt.legend()
        plt.title('Carrier Frequency Regression Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # 调制准确率
    plt.subplot(3, 2, 5)
    if mode in ['mod', 'joint'] and 'train_mod_acc' in history:
        plt.plot(epochs, history['train_mod_acc'], label='Train Acc')
        plt.plot(epochs, history['val_mod_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Modulation Accuracy')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    # DOA RMSE
    plt.subplot(3, 2, 6)
    if mode in ['doa', 'joint'] and 'val_doa_rmse' in history:
        plt.plot(epochs, history['val_doa_rmse'], label='Val DOA RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE (deg)')
        plt.legend()
        plt.title('DOA RMSE')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")

# ------------------------- 4. 训练、验证、测试函数 -------------------------
def train_epoch_jointnet_latent(model, loader, optimizer, loss_cls, loss_doa, device,
                                mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.train()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    pbar = tqdm(loader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        # 解包: cov, spectrum, angle, iq, mod, snr, fc (fc 虽然存在但不使用)
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred = torch.argmax(mod_out, dim=1)
            correct = (pred == mod_label).sum().item()
            total_correct += correct
            pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc


@torch.no_grad()
def eval_epoch_jointnet_latent(model, loader, loss_cls, loss_doa, device, DOA_grid,
                               mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.eval()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    total_doa_rmse, total_mae_doa = 0.0, 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            # doa_spectrum = doa_out
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(batch_size):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    rmse_doa = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    mae_doa = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc, rmse_doa, mae_doa


@torch.no_grad()
def test_jointnet_latent(model, loader, device, DOA_grid, mode='joint'):
    model.eval()
    total_correct, total_samples = 0, 0
    total_doa_rmse, total_mae_doa = 0.0, 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        if mode in ['mod', 'joint']:
            mod_out, doa_out = model(iq, cov)
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct
        else:
            _, doa_out = model(iq, cov)

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            # doa_spectrum = doa_out
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(iq.size(0)):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

        total_samples += iq.size(0)

    mod_acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    doa_rmse = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    doa_mae = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return mod_acc, doa_rmse, doa_mae


# ------------------------- 5. 绘图函数 -------------------------
def plot_jointnet_latent_history(history, save_dir, mode='joint'):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss')

    plt.subplot(3, 2, 2)
    if mode in ['mod', 'joint'] and 'train_cls_loss' in history:
        plt.plot(epochs, history['train_cls_loss'], label='Train Cls')
        plt.plot(epochs, history['val_cls_loss'], label='Val Cls')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Modulation Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 3)
    if mode in ['doa', 'joint'] and 'train_doa_loss' in history:
        plt.plot(epochs, history['train_doa_loss'], label='Train DOA')
        plt.plot(epochs, history['val_doa_loss'], label='Val DOA')
        plt.xlabel('Epoch')
        plt.ylabel('DOA Loss')
        plt.legend()
        plt.title('DOA Spectrum Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 4)
    if mode in ['mod', 'joint'] and 'train_mod_acc' in history:
        plt.plot(epochs, history['train_mod_acc'], label='Train Acc')
        plt.plot(epochs, history['val_mod_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Modulation Accuracy')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 5)
    if mode in ['doa', 'joint'] and 'val_doa_rmse' in history:
        plt.plot(epochs, history['val_doa_rmse'], label='Val DOA RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE (deg)')
        plt.legend()
        plt.title('DOA RMSE')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")


# 消融实验 没有这个隐藏的FC模块
def train_epoch_jointnet_wofreq(model, loader, optimizer, loss_cls, loss_doa, device,
                                mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.train()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    pbar = tqdm(loader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        _ = batch[2]
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred = torch.argmax(mod_out, dim=1)
            correct = (pred == mod_label).sum().item()
            total_correct += correct
            pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc


@torch.no_grad()
def eval_epoch_jointnet_wofreq(model, loader, loss_cls, loss_doa, device, DOA_grid,
                               mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.eval()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    total_doa_rmse, total_mae_doa = 0.0, 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(batch_size):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    rmse_doa = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    mae_doa = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc, rmse_doa, mae_doa


@torch.no_grad()
def test_jointnet_wofreq(model, loader, device, DOA_grid, mode='joint'):
    model.eval()
    total_correct, total_samples = 0, 0
    total_doa_rmse, total_mae_doa = 0.0, 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        if mode in ['mod', 'joint']:
            mod_out, doa_out = model(iq, cov)
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct
        else:
            _, doa_out = model(iq, cov)

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(iq.size(0)):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

        total_samples += iq.size(0)

    mod_acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    doa_rmse = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    doa_mae = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return mod_acc, doa_rmse, doa_mae


# ------------------------- 5. 绘图函数 -------------------------
def plot_jointnet_wofreq_history(history, save_dir, mode='joint'):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss')

    plt.subplot(3, 2, 2)
    if mode in ['mod', 'joint'] and 'train_cls_loss' in history:
        plt.plot(epochs, history['train_cls_loss'], label='Train Cls')
        plt.plot(epochs, history['val_cls_loss'], label='Val Cls')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Modulation Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 3)
    if mode in ['doa', 'joint'] and 'train_doa_loss' in history:
        plt.plot(epochs, history['train_doa_loss'], label='Train DOA')
        plt.plot(epochs, history['val_doa_loss'], label='Val DOA')
        plt.xlabel('Epoch')
        plt.ylabel('DOA Loss')
        plt.legend()
        plt.title('DOA Spectrum Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 4)
    if mode in ['mod', 'joint'] and 'train_mod_acc' in history:
        plt.plot(epochs, history['train_mod_acc'], label='Train Acc')
        plt.plot(epochs, history['val_mod_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Modulation Accuracy')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 5)
    if mode in ['doa', 'joint'] and 'val_doa_rmse' in history:
        plt.plot(epochs, history['val_doa_rmse'], label='Val DOA RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE (deg)')
        plt.legend()
        plt.title('DOA RMSE')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")

#验证MOD模型是否有存在的必要
# ------------------------- 4. 训练/验证/测试函数 -------------------------
def train_epoch_jointnet_doafreq(model, loader, optimizer, loss_cls, loss_doa, device,
                                 mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.train()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    pbar = tqdm(loader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        _ = batch[2]
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred = torch.argmax(mod_out, dim=1)
            correct = (pred == mod_label).sum().item()
            total_correct += correct
            pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc


@torch.no_grad()
def eval_epoch_jointnet_doafreq(model, loader, loss_cls, loss_doa, device, DOA_grid,
                                mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.eval()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    total_doa_rmse, total_mae_doa = 0.0, 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct

        if mode in ['doa', 'joint']:
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(batch_size):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    rmse_doa = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    mae_doa = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc, rmse_doa, mae_doa

#====验证是否用MOD支路
def train_epoch_jointnet_doafreq(model, loader, optimizer, loss_cls, loss_doa, device,
                                 mode='joint', lambda_cls=1.0, lambda_doa=1.0):
    model.train()
    total_loss, total_cls_loss, total_doa_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0
    pbar = tqdm(loader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        cov = batch[0].to(device)
        true_spectrum = batch[1].to(device)
        _ = batch[2]
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        mod_out, doa_out = model(iq, cov)

        loss = 0.0
        if mode in ['mod', 'joint']:
            loss_c = loss_cls(mod_out, mod_label)
            total_cls_loss += loss_c.item() * iq.size(0)
            loss += lambda_cls * loss_c
        if mode in ['doa', 'joint']:
            loss_d = loss_doa(doa_out, true_spectrum)
            total_doa_loss += loss_d.item() * iq.size(0)
            loss += lambda_doa * loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = iq.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        if mode in ['mod', 'joint']:
            pred = torch.argmax(mod_out, dim=1)
            correct = (pred == mod_label).sum().item()
            total_correct += correct
            pbar.set_postfix(loss=loss.item(), acc=correct/batch_size)
        else:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples if mode in ['mod', 'joint'] else 0.0
    avg_doa = total_doa_loss / total_samples if mode in ['doa', 'joint'] else 0.0
    acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    return avg_loss, avg_cls, avg_doa, acc

@torch.no_grad()
def test_jointnet_doafreq(model, loader, device, DOA_grid, mode='joint'):
    model.eval()
    total_correct, total_samples = 0, 0
    total_doa_rmse, total_mae_doa = 0.0, 0.0
    angle_grid_np = DOA_grid.cpu().numpy() if torch.is_tensor(DOA_grid) else DOA_grid

    for batch in loader:
        cov = batch[0].to(device)
        angle_true = batch[2].to(device)
        iq = batch[3].to(device)
        mod_label = batch[4].to(device)

        if mode in ['mod', 'joint']:
            mod_out, doa_out = model(iq, cov)
            pred_mod = torch.argmax(mod_out, dim=1)
            correct = (pred_mod == mod_label).sum().item()
            total_correct += correct
        else:
            _, doa_out = model(iq, cov)

        if mode in ['doa', 'joint']:
            # doa_spectrum = torch.sigmoid(doa_out)
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            for i in range(iq.size(0)):
                pred_angle, _ = findpeaks(doa_np[i], K=1, DOA=angle_grid_np)
                pred_angle = pred_angle[0]
                err = pred_angle - angle_true[i].item()
                total_doa_rmse += err ** 2
                total_mae_doa += abs(err)

        total_samples += iq.size(0)

    mod_acc = total_correct / total_samples * 100 if mode in ['mod', 'joint'] else 0.0
    doa_rmse = np.sqrt(total_doa_rmse / total_samples) if mode in ['doa', 'joint'] else 0.0
    doa_mae = total_mae_doa / total_samples if mode in ['doa', 'joint'] else 0.0
    return mod_acc, doa_rmse, doa_mae


# ------------------------- 5. 绘图函数 -------------------------
def plot_jointnet_doafreq_history(history, save_dir, mode='joint'):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss')

    plt.subplot(3, 2, 2)
    if mode in ['mod', 'joint'] and 'train_cls_loss' in history:
        plt.plot(epochs, history['train_cls_loss'], label='Train Cls')
        plt.plot(epochs, history['val_cls_loss'], label='Val Cls')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Modulation Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 3)
    if mode in ['doa', 'joint'] and 'train_doa_loss' in history:
        plt.plot(epochs, history['train_doa_loss'], label='Train DOA')
        plt.plot(epochs, history['val_doa_loss'], label='Val DOA')
        plt.xlabel('Epoch')
        plt.ylabel('DOA Loss')
        plt.legend()
        plt.title('DOA Spectrum Loss')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 4)
    if mode in ['mod', 'joint'] and 'train_mod_acc' in history:
        plt.plot(epochs, history['train_mod_acc'], label='Train Acc')
        plt.plot(epochs, history['val_mod_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Modulation Accuracy')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.subplot(3, 2, 5)
    if mode in ['doa', 'joint'] and 'val_doa_rmse' in history:
        plt.plot(epochs, history['val_doa_rmse'], label='Val DOA RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE (deg)')
        plt.legend()
        plt.title('DOA RMSE')
    else:
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")


