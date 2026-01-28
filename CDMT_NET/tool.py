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


# 数据集解包/可依自己数据集的生成习惯修改/
class H5DatasetMultiSNR(Dataset):
    def __init__(self, folder_path):
        self.RX_list = []
        self.XX_list = []
        self.S_label_list = []
        self.S_MR_label_list = []
        self.snr_label_list = []

        # 遍历 SNR 子文件夹
        for snr_dir in os.listdir(folder_path):
            if not snr_dir.startswith('SNR_'):
                continue

            snr_value = int(snr_dir.split('_')[-1])
            mat_file = os.path.join(folder_path, snr_dir, f'data_{snr_dir}.mat')

            with h5py.File(mat_file, 'r') as f:
                RX_raw = np.array(f['RX_local'])
                XX_raw = np.array(f['XX_local'])
                RX = RX_raw['real'] + 1j * RX_raw['imag']
                XX = XX_raw['real'] + 1j * XX_raw['imag']

                RX = RX.transpose(2, 0, 1)
                XX = XX.transpose(2, 1, 0)
                S_label = np.array(f['S_label_local']).transpose()
                S_MR_label = np.array(f['MR_label_local']).transpose()

            self.RX_list.append(RX.astype(np.complex64))
            self.XX_list.append(XX.astype(np.complex64))
            self.S_label_list.append(S_label.astype(np.float32))
            self.S_MR_label_list.append(S_MR_label.squeeze().astype(np.int64))
            self.snr_label_list.append(np.full(len(S_label), snr_value, dtype=np.int32))

        # 合并所有数据
        self.RX = np.concatenate(self.RX_list, axis=0)
        self.XX = np.concatenate(self.XX_list, axis=0)
        self.S_label = np.concatenate(self.S_label_list, axis=0)
        self.S_MR_label = np.concatenate(self.S_MR_label_list, axis=0)
        self.SNR_label = np.concatenate(self.snr_label_list, axis=0)

    def __len__(self):
        return len(self.RX)

    def __getitem__(self, idx):
        rx = self.RX[idx]
        xx = self.XX[idx]

        rx_data = np.stack([rx.real, rx.imag], axis=0)
        xx_data = np.stack([xx.real, xx.imag], axis=0)

        return (
            torch.tensor(rx_data, dtype=torch.float32),
            torch.tensor(xx_data, dtype=torch.float32),
            torch.tensor(self.S_MR_label[idx], dtype=torch.long),
            torch.tensor(self.S_label[idx], dtype=torch.float32),
            torch.tensor(self.SNR_label[idx], dtype=torch.int32)
            
        )


def plot_history(history, task, save_dir):
    """
    绘制训练历史曲线，支持：
    """
    # 创建保存目录（确保存在）
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------- 1. MOD任务（分类：损失+准确率） -----------------------
    if task in ['mod', 'both']:
        # 绘制MOD交叉熵损失曲线
        if 'mod' in history['train']['loss']:
            plt.figure(figsize=(8, 5))
            plt.plot(history['train']['loss']['mod'], label='Train Loss (CE)')
            plt.plot(history['val']['loss']['mod'], label='Val Loss (CE)')
            plt.title("MOD Cross-Entropy Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'mod_loss_curve.png'))
            plt.close()

        # 绘制MOD准确率曲线
        if 'mod' in history['train']['acc']:
            plt.figure(figsize=(8, 5))
            plt.plot(history['train']['acc']['mod'], label='Train Accuracy')
            plt.plot(history['val']['acc']['mod'], label='Val Accuracy')
            plt.title("MOD Accuracy Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'mod_acc_curve.png'))
            plt.close()

    # ----------------------- 2. DOA任务（分类：损失+准确率+RMSE） -----------------------
    if task in ['doa', 'both']:
        # 绘制DOA交叉熵损失曲线
        if 'doa' in history['train']['loss']:
            plt.figure(figsize=(8, 5))
            plt.plot(history['train']['loss']['doa'], label='Train Loss (CE)')
            plt.plot(history['val']['loss']['doa'], label='Val Loss (CE)')
            plt.title("DOA Cross-Entropy Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'doa_loss_curve.png'))
            plt.close()

        # 绘制DOA准确率曲线（分类任务核心指标）
        if 'doa' in history['train']['acc']:
            plt.figure(figsize=(8, 5))
            plt.plot(history['train']['acc']['doa'], label='Train Accuracy')
            plt.plot(history['val']['acc']['doa'], label='Val Accuracy')
            plt.title("DOA Accuracy Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'doa_acc_curve.png'))
            plt.close()

        # 绘制DOA RMSE曲线（角度误差辅助指标）
        if 'doa' in history['train']['rmse']:
            plt.figure(figsize=(8, 5))
            plt.plot(history['train']['rmse']['doa'], label='Train RMSE')
            plt.plot(history['val']['rmse']['doa'], label='Val RMSE')
            plt.title("DOA Angle RMSE Curve")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE (°)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'doa_rmse_curve.png'))
            plt.close()

    # ----------------------- 3. BOTH任务（总损失曲线） -----------------------
    if task == 'both':
        # 绘制总损失曲线（MOD+DOA加权和）
        if 'total' in history['train']['loss']:
            plt.figure(figsize=(8, 5))
            plt.plot(history['train']['loss']['total'], label='Train Total Loss')
            plt.plot(history['val']['loss']['total'], label='Val Total Loss')
            plt.title("Combined Total Loss (MOD + DOA) Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'both_total_loss_curve.png'))
            plt.close()

    print(f"History plots saved to {save_dir}")


# 角度转索引
def doa_angle_to_idx(angle_array):
    return angle_array + 40

def doa_idx_to_angle(idx_array):
    return idx_array - 40  # idx ∈ [0, 80] → angle ∈ [-40, 40]

# 角度转换部分有点绕注意逻辑对应
def evaluate_by_snr(model, test_loader, device, task, result_dir):
    model.eval()
    os.makedirs(result_dir, exist_ok=True)

    # ----------- 存储容器 ----------- #
    # Modulation
    mod_true, mod_pred, mod_snr = [], [], []

    # DOA
    true_doa_angle, pred_doa_angle, snr_labels = [], [], []

    mod_labels = [
        "FSK", "BPSK", "LFM", "FRANK",
        "P1", "P2", "P3", "P4",
        "T1", "T2", "T3", "T4"
    ]

    with torch.no_grad():
        for sample in test_loader:
            doa_input = sample[0].to(device)
            mod_input = sample[1].to(device)
            true_mod_batch = sample[2].cpu().numpy()
            true_doa_batch = sample[3].cpu().numpy()
            snr_batch = sample[4].cpu().numpy()

            mod_out, doa_out, result_mod, result_doa = model(
                mod_input=mod_input,
                doa_input=doa_input,
                task=task
            )

            # -------- Modulation -------- #
            if task in ['mod', 'both']:
                mod_pred_batch = result_mod.cpu().numpy()
                mod_true.extend(true_mod_batch)
                mod_pred.extend(mod_pred_batch)
                mod_snr.extend(snr_batch)

            # -------- DOA -------- #
            if task in ['doa', 'both']:
                pred_doa_batch = result_doa.cpu().numpy()

                # 转换角度（idx <-> angle）
                if np.any(true_doa_batch < 0):
                    true_doa_idx = doa_angle_to_idx(true_doa_batch)
                else:
                    true_doa_idx = true_doa_batch
                pred_doa_idx = pred_doa_batch

                true_angle_batch = doa_idx_to_angle(true_doa_idx)
                pred_angle_batch = doa_idx_to_angle(pred_doa_idx)

                true_doa_angle.extend(true_angle_batch)
                pred_doa_angle.extend(pred_angle_batch)
                snr_labels.extend(snr_batch)

    # ------------------- 根据任务选择 unique_snr ------------------- #
    if task == 'mod':
        unique_snr = np.sort(np.unique(mod_snr)) if len(mod_snr) > 0 else []
    elif task == 'doa':
        unique_snr = np.sort(np.unique(snr_labels)) if len(snr_labels) > 0 else []
    elif task == 'both':
        # 两个都有的话，用 DOA 的 snr_labels（和 mod_snr 应该一致）
        unique_snr = np.sort(np.unique(snr_labels)) if len(snr_labels) > 0 else []
    else:
        unique_snr = []

    # ------------------- Modulation 评估 ------------------- #
    if task in ['mod', 'both'] and len(mod_true) > 0:
        mod_true = np.array(mod_true)
        mod_pred = np.array(mod_pred)
        mod_snr = np.array(mod_snr)

        # 0 dB 混淆矩阵
        zero_db_mask = (mod_snr == 0)
        if np.sum(zero_db_mask) > 0:
            cm = confusion_matrix(mod_true[zero_db_mask], mod_pred[zero_db_mask], labels=np.arange(12))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                        xticklabels=mod_labels, yticklabels=mod_labels,
                        cbar_kws={'label': 'Percentage (%)'})
            plt.xlabel('Predicted Modulation')
            plt.ylabel('True Modulation')
            plt.title('Confusion Matrix (0 dB SNR)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'mod_confusion_matrix_0db.png'), dpi=300)
            plt.close()
            print("0dB SNR modulation confusion matrix saved.")

        # Accuracy vs SNR
        mod_acc_per_snr = []
        for snr in unique_snr:
            mask = (mod_snr == snr)
            if np.sum(mask) == 0:
                continue
            acc = np.mean(mod_pred[mask] == mod_true[mask])
            mod_acc_per_snr.append(acc)

        plt.figure(figsize=(10, 5))
        plt.plot(unique_snr, mod_acc_per_snr, marker='^', color='g', label='Modulation Accuracy')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        plt.title('Modulation Accuracy vs SNR')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'modulation_accuracy_vs_snr.png'))
        plt.close()

        # 每类调制 Accuracy vs SNR
        num_classes = len(mod_labels)
        class_acc_dict = {label: [] for label in mod_labels}
        for snr in unique_snr:
            mask = (mod_snr == snr)
            for i, label in enumerate(mod_labels):
                class_mask = mask & (mod_true == i)
                acc = np.mean(mod_pred[class_mask] == mod_true[class_mask]) if np.sum(class_mask) > 0 else np.nan
                class_acc_dict[label].append(acc)

        plt.figure(figsize=(14, 8))
        colors = plt.cm.get_cmap('tab20', num_classes)
        for i, label in enumerate(mod_labels):
            plt.plot(unique_snr, class_acc_dict[label], marker='o', label=label, color=colors(i))
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        plt.title('Modulation Accuracy per Class vs SNR')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(result_dir, 'modulation_accuracy_per_class_vs_snr.png'))
        plt.close()

        # 保存结果到 txt
        with open(os.path.join(result_dir, 'modulation_accuracy.txt'), 'w') as f:
            f.write("SNR(dB)\tModulation_Accuracy\n")
            for snr, acc in zip(unique_snr, mod_acc_per_snr):
                f.write(f"{snr}\t{acc:.4f}\n")

        with open(os.path.join(result_dir, 'modulation_accuracy_per_class.txt'), 'w') as f:
            header = "SNR(dB)\t" + "\t".join(mod_labels) + "\n"
            f.write(header)
            for i, snr in enumerate(unique_snr):
                accs = [class_acc_dict[label][i] if not np.isnan(class_acc_dict[label][i]) else -1 for label in mod_labels]
                accs_str = "\t".join(f"{acc:.4f}" if acc >= 0 else "NaN" for acc in accs)
                f.write(f"{snr}\t{accs_str}\n")

    # ------------------- DOA 评估 ------------------- #
    if task in ['doa', 'both'] and len(unique_snr) > 0:
        true_doa_angle = np.array(true_doa_angle)
        pred_doa_angle = np.array(pred_doa_angle)
        snr_labels = np.array(snr_labels)

        doa_rmse_per_snr = []
        for snr in unique_snr:
            mask = (snr_labels == snr)
            if np.sum(mask) == 0:
                continue
            rmse = np.sqrt(mean_squared_error(true_doa_angle[mask], pred_doa_angle[mask]))
            doa_rmse_per_snr.append(rmse)
            print(f"SNR={snr} dB, DOA_RMSE={rmse:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(unique_snr, doa_rmse_per_snr, marker='s', color='r', label='DOA RMSE')
        plt.xlabel('SNR (dB)')
        plt.ylabel('RMSE (°)')
        plt.title('DOA Angle RMSE vs SNR')
        plt.grid(True)
        plt.legend()
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(result_dir, 'doa_rmse_vs_snr.png'))
        plt.close()

        # 保存 txt
        with open(os.path.join(result_dir, 'doa_rmse.txt'), 'w') as f:
            f.write("SNR(dB)\tDOA_RMSE\n")
            for snr, rmse in zip(unique_snr, doa_rmse_per_snr):
                f.write(f"{snr}\t{rmse:.4f}\n")

    print(f"SNR-based evaluation completed. Results saved to {result_dir}")

