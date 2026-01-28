import os
import argparse

parser = argparse.ArgumentParser(description="CDMT_Net")
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="2", help='GPU id')
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
from model import * 
from tool import * 
from param import * 

# 训练主函数
def main(task='both'):
    result_dir = f'/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/test_{task}_0128' #保存路径
    os.makedirs(result_dir, exist_ok=True)
    file_path = '/home/sp432sl/ZQ/DOA_regcition/multi_snr_dataset' #数据集导入路径

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------------- 1. 加载数据与按 SNR 分组划分 ----------------------- #
    dataset = H5DatasetMultiSNR(file_path)

    def get_snr_labels(dataset):
        snrs = []
        for i in range(len(dataset)):
            _, _, _, _, snr = dataset[i]
            snrs.append(snr.item())
        return snrs

    snrs = get_snr_labels(dataset)

    from collections import defaultdict
    from sklearn.model_selection import train_test_split

    snr_to_indices = defaultdict(list)
    for idx, snr in enumerate(snrs):
        snr_to_indices[snr].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    print("\nSNR-wise sample counts (train/val/test):")
    for snr, indices in snr_to_indices.items():
        if len(indices) < 5:
            print(f"SNR {snr} has too few samples ({len(indices)}), assigning all to train.")
            train_idx.extend(indices)
            continue

        train_split, temp_split = train_test_split(indices, test_size=0.4, random_state=42, shuffle=True)
        val_split, test_split = train_test_split(temp_split, test_size=0.5, random_state=42, shuffle=True)

        train_idx.extend(train_split)
        val_idx.extend(val_split)
        test_idx.extend(test_split)

        print(f"SNR {snr}: {len(train_split)} / {len(val_split)} / {len(test_split)}")

    print(f"Total: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ----------------------- 2. 构建数据集和加载器 ----------------------- #
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # ----------------------- 3. 初始化模型与优化器 ----------------------- #
    model = CDMT_Net(num_classes_mod=12, num_classes_doa=81).to(device)

    criterion_mod = nn.CrossEntropyLoss() if task in ['mod', 'both'] else None
    criterion_doa = nn.CrossEntropyLoss() if task in ['doa', 'both'] else None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7)

    # ----------------------- 4. 训练历史记录初始化 ----------------------- #
    num_epochs = 1
    patience = 15
    early_stop_counter = 0
    best_val_loss = float('inf')
    model_path = os.path.join(result_dir, f'best_model_{task}.pth')

    train_history = {'loss': {}, 'acc': {}, 'rmse': {}}
    val_history = {'loss': {}, 'acc': {}, 'rmse': {}}

    if task in ['mod', 'both']:
        train_history['loss']['mod'] = []
        train_history['acc']['mod'] = []
        val_history['loss']['mod'] = []
        val_history['acc']['mod'] = []

    if task in ['doa', 'both']:
        train_history['loss']['doa'] = []
        train_history['acc']['doa'] = []
        train_history['rmse']['doa'] = []
        val_history['loss']['doa'] = []
        val_history['acc']['doa'] = []
        val_history['rmse']['doa'] = []

    if task == 'both':
        train_history['loss']['total'] = []
        val_history['loss']['total'] = []

    # ----------------------- 5. 训练主循环 ----------------------- #
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion_mod, criterion_doa, device, task=task)
        val_metrics = valid_epoch(model, val_loader, criterion_mod, criterion_doa, device, task=task)

        if task in ['mod', 'both']:
            train_history['loss']['mod'].append(train_metrics['loss']['mod'])
            train_history['acc']['mod'].append(train_metrics['acc']['mod'])
            val_history['loss']['mod'].append(val_metrics['loss']['mod'])
            val_history['acc']['mod'].append(val_metrics['acc']['mod'])

        if task in ['doa', 'both']:
            train_history['loss']['doa'].append(train_metrics['loss']['doa'])
            train_history['acc']['doa'].append(train_metrics['acc']['doa'])
            train_history['rmse']['doa'].append(train_metrics['rmse']['doa'])
            val_history['loss']['doa'].append(val_metrics['loss']['doa'])
            val_history['acc']['doa'].append(val_metrics['acc']['doa'])
            val_history['rmse']['doa'].append(val_metrics['rmse']['doa'])

        if task == 'both':
            total_train_loss = train_metrics['loss']['mod'] * mod_weight + train_metrics['loss']['doa'] * doa_weight
            total_val_loss = val_metrics['loss']['mod'] * mod_weight + val_metrics['loss']['doa'] * doa_weight
            train_history['loss']['total'].append(total_train_loss)
            val_history['loss']['total'].append(total_val_loss)
        else:
            total_val_loss = val_metrics['loss']['mod'] if task == 'mod' else val_metrics['loss']['doa']

        scheduler.step(total_val_loss)

        # 日志输出
        print(f"Train Loss - Mod: {train_metrics['loss'].get('mod', 0):.6f}, "
              f"DOA Loss: {train_metrics['loss'].get('doa', 0):.6f}")
        print(f"Val   Loss - Mod: {val_metrics['loss'].get('mod', 0):.6f}, "
              f"DOA Loss: {val_metrics['loss'].get('doa', 0):.6f}")

        if task in ['mod', 'both']:
            print(f"Train Acc  - Mod: {train_metrics['acc'].get('mod', 0):.6f}")
            print(f"Val   Acc  - Mod: {val_metrics['acc'].get('mod', 0):.6f}")

        if task in ['doa', 'both']:
            print(f"Train Acc  - DOA: {train_metrics['acc'].get('doa', 0):.6f}, "
                  f"DOA RMSE: {train_metrics['rmse'].get('doa', 0):.6f}°")
            print(f"Val   Acc  - DOA: {val_metrics['acc'].get('doa', 0):.6f}, "
                  f"DOA RMSE: {val_metrics['rmse'].get('doa', 0):.6f}°")

        if task == 'both':
            print(f"Train Total Loss: {total_train_loss:.6f}")
            print(f"Val   Total Loss: {total_val_loss:.6f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
            print(f"Saved best model with val_loss: {best_val_loss:.6f}")
        else:
            early_stop_counter += 1
            print(f"No improvement, early_stop_counter = {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # ----------------------- 6. 保存训练历史与评估 ----------------------- #
    with open(os.path.join(result_dir, f'train_val_history_{task}.pkl'), 'wb') as f:
        pickle.dump({'train': train_history, 'val': val_history}, f)

    model.load_state_dict(torch.load(model_path))

    if task == 'mod':
        plot_history({'train': train_history, 'val': val_history}, task='mod', save_dir=result_dir)
    elif task == 'doa':
        plot_history({'train': train_history, 'val': val_history}, task='doa', save_dir=result_dir)
    else:
        plot_history({'train': train_history, 'val': val_history}, task='mod', save_dir=result_dir)
        plot_history({'train': train_history, 'val': val_history}, task='doa', save_dir=result_dir)
        plot_history({'train': train_history, 'val': val_history}, task='both', save_dir=result_dir)

    evaluate_by_snr(model, test_loader, device, task=task, result_dir=result_dir)
    print("\nTraining completed and SNR evaluation done.")

# ----------------------- 主程序 ----------------------- #
if __name__ == "__main__":
    main(task='both')  # 根据需求调整 task：'mod'/'doa'/'both'

