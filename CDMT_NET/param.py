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

#训练测试函数
def train_epoch(model, train_loader, optimizer, criterion_mod, criterion_doa, device, task='both'):
    model.train()
    running_loss_mod = 0.0
    running_loss_doa = 0.0
    correct_mod = 0
    correct_doa = 0
    total_samples = 0
    doa_squared_error = 0.0  # 用于计算 RMSE

    progress_bar = tqdm(train_loader, desc="Train Batches", leave=False, ncols=120)

    for inputs_rx, inputs_xx, labels_mod, labels_doa, labels_snr in progress_bar:
        inputs_xx = inputs_xx.to(device)
        inputs_rx = inputs_rx.to(device)
        labels_mod = labels_mod.to(device).long()

        # 对 DOA 标签进行索引映射：[-40, 40] -> [0, 80]
        labels_doa = labels_doa.to(device).long() + 40
        labels_doa = labels_doa.squeeze(1)

        optimizer.zero_grad()

        # 前向传播
        if task == 'mod':
            outputs_mod, _, _, _ = model(mod_input=inputs_xx, task='mod')
            result_mod = torch.argmax(outputs_mod, dim=1)
        elif task == 'doa':
            _, outputs_doa, _, _ = model(doa_input=inputs_rx, task='doa')
            result_doa = torch.argmax(outputs_doa, dim=1)
        else:  # both
            outputs_mod, outputs_doa, _, _ = model(mod_input=inputs_xx, doa_input=inputs_rx, task='both')
            result_mod = torch.argmax(outputs_mod, dim=1)
            result_doa = torch.argmax(outputs_doa, dim=1)

        loss = 0
        batch_size = inputs_xx.size(0)
        total_samples += batch_size

        mod_weight = 1.0
        doa_weight = 0.01

        # 分类损失 MOD
        if task in ['mod', 'both']:
            loss_mod = criterion_mod(outputs_mod, labels_mod)
            running_loss_mod += loss_mod.item() * batch_size
        else:
            loss_mod = 0

        # 分类损失 DOA
        if task in ['doa', 'both']:
            loss_doa = criterion_doa(outputs_doa, labels_doa)
            running_loss_doa += loss_doa.item() * batch_size
        else:
            loss_doa = 0

        # 合并损失
        if task == 'both':
            loss = loss_mod * mod_weight + loss_doa * doa_weight
        elif task == 'mod':
            loss = loss_mod
        else:
            loss = loss_doa

        # 反向传播
        loss.backward()
        optimizer.step()

        # 分类正确数（MOD）
        if task in ['mod', 'both']:
            correct_mod += (result_mod == labels_mod).sum().item()

        # DOA分类准确数 + RMSE计算
        if task in ['doa', 'both']:
            correct_doa += (result_doa == labels_doa).sum().item()

            # 计算 RMSE：将索引映射回角度值：0~80 -> -40~40
            pred_angle = result_doa - 40
            true_angle = labels_doa - 40
            doa_squared_error += torch.sum((pred_angle.float() - true_angle.float()) ** 2).item()

        # 日志显示
        current_loss_mod = running_loss_mod / total_samples if task in ['mod', 'both'] else 0
        current_loss_doa = running_loss_doa / total_samples if task in ['doa', 'both'] else 0
        current_acc_mod = correct_mod / total_samples if task in ['mod', 'both'] else 0
        current_acc_doa = correct_doa / total_samples if task in ['doa', 'both'] else 0
        current_rmse_doa = (doa_squared_error / total_samples) ** 0.5 if task in ['doa', 'both'] else 0

        progress_bar.set_postfix(
            mod_loss=f"{current_loss_mod:.4f}" if task in ['mod', 'both'] else "N/A",
            doa_loss=f"{current_loss_doa:.4f}" if task in ['doa', 'both'] else "N/A",
            mod_acc=f"{current_acc_mod:.4f}" if task in ['mod', 'both'] else "N/A",
            doa_acc=f"{current_acc_doa:.4f}" if task in ['doa', 'both'] else "N/A",
            doa_rmse=f"{current_rmse_doa:.2f}" if task in ['doa', 'both'] else "N/A"
        )

    # 平均损失和准确率
    avg_loss_mod = running_loss_mod / total_samples if task in ['mod', 'both'] else 0
    avg_loss_doa = running_loss_doa / total_samples if task in ['doa', 'both'] else 0
    acc_mod = correct_mod / total_samples if task in ['mod', 'both'] else 0
    acc_doa = correct_doa / total_samples if task in ['doa', 'both'] else 0
    doa_rmse = (doa_squared_error / total_samples) ** 0.5 if task in ['doa', 'both'] else 0

    return {
        'loss': {
            'mod': avg_loss_mod,
            'doa': avg_loss_doa
        },
        'acc': {
            'mod': acc_mod,
            'doa': acc_doa
        },
        'rmse': {
            'doa': doa_rmse
        }
    }

def valid_epoch(model, val_loader, criterion_mod, criterion_doa, device, task='both'):
    model.eval()
    running_loss_mod = 0.0
    running_loss_doa = 0.0
    correct_mod = 0
    correct_doa = 0
    doa_squared_error = 0.0
    total_samples_mod = 0
    total_samples_doa = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation Batches", leave=False, ncols=120)

        for inputs_rx, inputs_xx, labels_mod, labels_doa, labels_snr in progress_bar:
            inputs_xx = inputs_xx.to(device)
            inputs_rx = inputs_rx.to(device)
            labels_mod = labels_mod.to(device).long()

            # DOA 标签从 [-40, 40] 转换为 [0, 80]
            labels_doa = labels_doa.to(device).long() + 40
            labels_doa = labels_doa.squeeze(1)

            if task == 'mod':
                outputs_mod, _, _, _ = model(mod_input=inputs_xx, task='mod')
                batch_size_mod = inputs_xx.size(0)
                total_samples_mod += batch_size_mod

                loss_mod = criterion_mod(outputs_mod, labels_mod)
                running_loss_mod += loss_mod.item() * batch_size_mod

                preds_mod = torch.argmax(outputs_mod, dim=1)
                correct_mod += (preds_mod == labels_mod).sum().item()

            elif task == 'doa':
                _, outputs_doa, _, _ = model(doa_input=inputs_rx, task='doa')
                batch_size_doa = inputs_rx.size(0)
                total_samples_doa += batch_size_doa

                loss_doa = criterion_doa(outputs_doa, labels_doa)
                running_loss_doa += loss_doa.item() * batch_size_doa

                preds_doa = torch.argmax(outputs_doa, dim=1)
                correct_doa += (preds_doa == labels_doa).sum().item()

                # RMSE
                pred_angle = preds_doa - 40
                true_angle = labels_doa - 40
                doa_squared_error += torch.sum((pred_angle.float() - true_angle.float()) ** 2).item()

            else:  # both
                outputs_mod, outputs_doa, _, _ = model(mod_input=inputs_xx, doa_input=inputs_rx, task='both')

                batch_size_mod = inputs_xx.size(0)
                batch_size_doa = inputs_rx.size(0)
                total_samples_mod += batch_size_mod
                total_samples_doa += batch_size_doa

                # MOD
                loss_mod = criterion_mod(outputs_mod, labels_mod)
                running_loss_mod += loss_mod.item() * batch_size_mod
                preds_mod = torch.argmax(outputs_mod, dim=1)
                correct_mod += (preds_mod == labels_mod).sum().item()

                # DOA
                loss_doa = criterion_doa(outputs_doa, labels_doa)
                running_loss_doa += loss_doa.item() * batch_size_doa
                preds_doa = torch.argmax(outputs_doa, dim=1)
                correct_doa += (preds_doa == labels_doa).sum().item()

                pred_angle = preds_doa - 40
                true_angle = labels_doa - 40
                doa_squared_error += torch.sum((pred_angle.float() - true_angle.float()) ** 2).item()

            # 显示日志
            current_loss_mod = running_loss_mod / total_samples_mod if total_samples_mod > 0 else 0
            current_loss_doa = running_loss_doa / total_samples_doa if total_samples_doa > 0 else 0
            current_acc_mod = correct_mod / total_samples_mod if total_samples_mod > 0 else 0
            current_acc_doa = correct_doa / total_samples_doa if total_samples_doa > 0 else 0
            current_rmse_doa = (doa_squared_error / total_samples_doa) ** 0.5 if total_samples_doa > 0 else 0

            progress_bar.set_postfix(
                mod_loss=f"{current_loss_mod:.4f}" if task in ['mod', 'both'] else "N/A",
                doa_loss=f"{current_loss_doa:.4f}" if task in ['doa', 'both'] else "N/A",
                mod_acc=f"{current_acc_mod:.4f}" if task in ['mod', 'both'] else "N/A",
                doa_acc=f"{current_acc_doa:.4f}" if task in ['doa', 'both'] else "N/A",
                doa_rmse=f"{current_rmse_doa:.2f}" if task in ['doa', 'both'] else "N/A"
            )

    avg_loss_mod = running_loss_mod / total_samples_mod if total_samples_mod > 0 else 0
    avg_loss_doa = running_loss_doa / total_samples_doa if total_samples_doa > 0 else 0
    avg_acc_mod = correct_mod / total_samples_mod if total_samples_mod > 0 else 0
    avg_acc_doa = correct_doa / total_samples_doa if total_samples_doa > 0 else 0
    doa_rmse = (doa_squared_error / total_samples_doa) ** 0.5 if total_samples_doa > 0 else 0

    return {
        'loss': {
            'mod': avg_loss_mod,
            'doa': avg_loss_doa
        },
        'acc': {
            'mod': avg_acc_mod,
            'doa': avg_acc_doa
        },
        'rmse': {
            'doa': doa_rmse
        }
    }
