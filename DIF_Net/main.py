import os
import argparse


parser = argparse.ArgumentParser(description="DIF_Net")
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
from torch.optim.lr_scheduler import CosineAnnealingLR


# 其他依赖
import matplotlib.pyplot as plt
from scipy.io import loadmat
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
from thop import profile, clever_format

from unit import * 
from model import * 
from tool import * 
from param import * 
from loss import * 
from JCMR import * 

# 用的含有十二种调制类型和方位角信息的数据集
# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def main_jointnet_latent(mode='joint'):   # mode: 'mod', 'doa', 'joint'
    # 路径配置（请根据实际情况修改）
    file_path = "/home/sp432sl/ZQ/DOA_regcition_0908/DIF-Net/dataset" #DIF-Net
    result_dir = f"/home/sp432sl/ZQ/DOA_regcition_0908/DIF-Net_0718/model2/joint_latent_AdaptiveFusionnotrans{mode}" 
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Training mode: {mode}")

    # 数据集（需要返回 SNR 用于分层划分）
    dataset = H5MultiTaskDataset(
        folder_path=file_path,
        return_cov=True, return_spectrum=True, return_angle=True,
        return_iq=True, return_mod=True, return_snr=True,   # 必须为 True 以便分层
        return_fc=False,  # 不使用载频标签
        fc_norm='logminmax', iq_shape='2D'
    )

    # sample = dataset[0]


    # cov = sample[0]
    # spectrum = sample[1]
    # angle = sample[2]
    # iq = sample[3]
    # mod = sample[4]


    # print("===== Dataset Check =====")

    # print("Spectrum min:", spectrum.min())
    # print("Spectrum max:", spectrum.max())

    # print("Spectrum has NaN:",
    #   torch.isnan(spectrum).any())

    # print("Spectrum has Inf:",
    #   torch.isinf(spectrum).any())


    # print("IQ has NaN:",
    #   torch.isnan(iq).any())

    # print("IQ has Inf:",
    #   torch.isinf(iq).any())


    # print("Cov has NaN:",
    #   torch.isnan(cov).any())

    # print("Cov has Inf:",
    #   torch.isinf(cov).any())


    # print("========================")
    
    # DOA 网格
    angle_min, angle_max, step = -60, 60, 1
    DOA_grid = np.linspace(angle_min, angle_max, int((angle_max - angle_min)/step) + 1)
    num_doa_classes = len(DOA_grid)

    # 按 SNR 分层划分
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

    model = JointNet_AdaptiveFusion(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes,
        embed_dim=64,
        dropout_rate=0.3
    ).to(device) 

    model.eval()  # 设置为评估模式

    # 修正：mod_input 最后一维为 1
    dummy_input = torch.randn(1, 2, 768, 1).to(device)
    dummy_cov = torch.randn(1, 2, 8, 8).to(device)   # DOA 输入正确

    # 先验证前向是否正常
    with torch.no_grad():
        out1, out2 = model(dummy_input, dummy_cov)
    print("✓ 前向推理正常，输出形状：", out1.shape, out2.shape)

    # 1. 参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. FLOPs（单位：M）
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_input, dummy_cov), verbose=False)
        mflops = flops / 1e6
    except Exception as e:
        print(f"thop 统计失败: {e}")
        mflops = 0.0

    # 3. 显存占用（MB）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input, dummy_cov)
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        mem_mb = 0

    # 4. Latency（ms）
    if torch.cuda.is_available():
        # 预热
        for _ in range(10):
            _ = model(dummy_input, dummy_cov)
        num_runs = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        for _ in range(num_runs):
            _ = model(dummy_input, dummy_cov)
        end_event.record()
        torch.cuda.synchronize(device)
        latency_ms = start_event.elapsed_time(end_event) / num_runs
    else:
        for _ in range(10):
            _ = model(dummy_input, dummy_cov)
        num_runs = 100
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input, dummy_cov)
        latency_ms = (time.perf_counter() - start) * 1000 / num_runs

    # 5. 吞吐量（MFLOPs/s）
    if mflops > 0 and latency_ms > 0:
        throughput_mflops = mflops / (latency_ms / 1000.0)
    else:
        throughput_mflops = 0.0

    print(f"  Params (M): {round(total_params, 3)}")
    print(f"  FLOPs (M): {round(mflops, 3)}")
    print(f"  GPU Memory (MB): {round(mem_mb, 2)}")
    print(f"  Latency (ms): {round(latency_ms, 3)}")
    print(f"  Throughput (MFLOPs/s): {round(throughput_mflops, 3)}")


    # 损失函数
    loss_cls = nn.CrossEntropyLoss()
    # loss_doa = nn.MSELoss()
    loss_doa = nn.BCEWithLogitsLoss()

    # 损失权重（可调）
    lambda_cls = 1.0
    lambda_doa = 1.0
    print(f"Loss weights: cls={lambda_cls}, doa={lambda_doa}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)

    # 历史记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_doa_loss': [], 'val_doa_loss': [],
        'train_mod_acc': [], 'val_mod_acc': [],
        'val_doa_rmse': []
    }

    best_val_metric = float('inf') if mode != 'mod' else -float('inf')
    patience, early_stop_cnt = 10, 0
    model_path = os.path.join(result_dir, f'best_model_{mode}.pth')
    num_epochs = 200

    print("\n" + "="*50)
    print(f"Starting {mode.upper()} Training with JointNet_LF (latent frequency)")
    print("="*50)

    for epoch in range(num_epochs):
        train_loss, train_cls, train_doa, train_acc = train_epoch_jointnet_latent(
            model, train_loader, optimizer, loss_cls, loss_doa, device,
            mode=mode, lambda_cls=lambda_cls, lambda_doa=lambda_doa
        )
        val_loss, val_cls, val_doa, val_acc, val_rmse, val_mae = eval_epoch_jointnet_latent(
            model, val_loader, loss_cls, loss_doa, device, DOA_grid,
            mode=mode, lambda_cls=lambda_cls, lambda_doa=lambda_doa
        )
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if mode in ['mod', 'joint']:
            history['train_cls_loss'].append(train_cls)
            history['val_cls_loss'].append(val_cls)
            history['train_mod_acc'].append(train_acc)
            history['val_mod_acc'].append(val_acc)
        if mode in ['doa', 'joint']:
            history['train_doa_loss'].append(train_doa)
            history['val_doa_loss'].append(val_doa)
            history['val_doa_rmse'].append(val_rmse)

        # 打印
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        if mode in ['mod', 'joint']:
            print(f"  Train - Loss: {train_loss:.6f} (Cls: {train_cls:.6f}) Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.6f} (Cls: {val_cls:.6f}) Acc: {val_acc:.2f}%")
        if mode in ['doa', 'joint']:
            print(f"  DOA - Train Loss: {train_doa:.6f}, Val Loss: {val_doa:.6f}")
            print(f"        Val DOA RMSE: {val_rmse:.4f}°, MAE: {val_mae:.4f}°")

        # 保存最佳模型
        if mode == 'mod':
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                torch.save(model.state_dict(), model_path)
                early_stop_cnt = 0
                print(f"  ✓ Saved best model (Val Acc: {best_val_metric:.2f}%)")
            else:
                early_stop_cnt += 1
        else:  # doa or joint
            if val_rmse < best_val_metric:
                best_val_metric = val_rmse
                torch.save(model.state_dict(), model_path)
                early_stop_cnt = 0
                print(f"  ✓ Saved best model (Val DOA RMSE: {best_val_metric:.4f}°)")
            else:
                early_stop_cnt += 1

        if early_stop_cnt >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 保存历史与绘图
    with open(os.path.join(result_dir, f'training_history_{mode}.pkl'), 'wb') as f:
        pickle.dump(history, f)
    plot_jointnet_latent_history(history, result_dir, mode)

    # 测试最佳模型
    model.load_state_dict(torch.load(model_path))
    test_acc, test_rmse, test_mae = test_jointnet_latent(model, test_loader, device, DOA_grid, mode)
    print("\n========== Final Test Results ==========")
    if mode in ['mod', 'joint']:
        print(f"Test Modulation Accuracy: {test_acc:.2f}%")
    if mode in ['doa', 'joint']:
        print(f"Test DOA RMSE: {test_rmse:.4f}°, MAE: {test_mae:.4f}°")

    print(f"\nAll results saved to {result_dir}")


# 测试联合模型 JointNet_LF（隐式频率）
def test_joint_latent_full():
    # 测试隐式频率联合模型 JointNet_LF（调制识别 + DOA 估计）
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/dataset/testdata_modulation_suijifc'
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/dataset/test_modulation_suijifc-1616'
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/dataset/test_modulation_suijifc-1414'
    # 用的测试的数据集
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0710/dataset/M8newdataset'
    # snap长度不一致 角度为10.16
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/dataset/test_modulation_fc1.25'
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/different_snap_dataset/10/768'

    #M=7
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/testdataset'

    # concat
    # model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0710/model/joint_latent_joint/best_model_joint.pth'
    # nocross
    # model_path ='/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0710/model/joint_latent_woFreqjoint/best_model_joint.pth'
     # adaptive
    # model_path ='/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/model2/joint_latent_AdaptiveFusionjoint/best_model_joint.pth'
    #JointNet_AdaptiveFusion_woTrans
    model_path ='/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/model2/joint_latent_AdaptiveFusionnotransjoint/best_model_joint.pth'
    

    # model_path='/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0710/model/joint_latentnoglobalcontex_joint/best_model_joint.pth'
    # result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/CDMT_NET/model2/jointmodel/joint_prefcdoa/joint_latentnoglobalcontex_joint/result'
    # os.makedirs(result_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DOA 角度网格
    angle_min, angle_max, step = -60, 60, 1
    DOA_grid = np.linspace(angle_min, angle_max, int((angle_max - angle_min)/step) + 1)
    num_doa_classes = len(DOA_grid)

    # 调制类型名称
    mod_names = ["FSK4", "LFM", "BPSK", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    # 加载测试数据集（需要调制标签和SNR，不需要外部载频标签）
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True, return_spectrum=True, return_angle=True,
        return_iq=True, return_mod=True, return_snr=True,
        return_fc=False,           # 隐式频率模型不需要外部fc标签
        fc_norm='logminmax', iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 加载模型
    # model = JointNet_LF(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    #nocross
    # model = JointNet_LF_nocross(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    #AdaptiveFusion wofreq newcdmt
    model = JointNet_AdaptiveFusion(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes,
        embed_dim=64,
        dropout_rate=0.3
    ).to(device)
    #AdaptiveFusion woCET
    # model = JointNet_AdaptiveFusion_woTrans(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)
    
    #noglobal
    # model = JointNet_LF_woFreq(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    # model = JointNet_LF_Ablation_Freq2DOA(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    # model = JointNet_LFV2(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)
    # model = JointNet_LFV3(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    # model = JointNet_LF_Ablation_Freq2DOA(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)


    # model = JointNet_LF_Stat(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)
    # 应用FILM层
    # model = JointNet_LF_FiLM(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    # 应用SE模块
    # model = JointNet_LF_SE(
    #     num_classes_mod=12,
    #     num_classes_doa=num_doa_classes,
    #     embed_dim=64,
    #     dropout_rate=0.3
    # ).to(device)

    # model = JointNet_JCAMR(
    #     mod_input_channels=2,
    #     doa_input_channels=2,
    #     mod_tcn_channels=[64, 128, 128],
    #     doa_tcn_channels=[64, 128, 128],
    #     tcn_kernel_size=3,
    #     tcn_dropout=0.2,
    #     fusion_layers=2,
    #     fusion_dropout=0.3,
    #     num_classes_mod=12,
    #     num_classes_doa=121,
    #     embed_dim=128
    # ).to(device)

        # 计算 FLOPs 和参数量
    dummy_iq = torch.randn(1, 2, 1024, 1).to(device)
    dummy_cov = torch.randn(1, 2, 8, 8).to(device)   # 根据实际 M 调整
    flops, params = profile(model, inputs=(dummy_iq, dummy_cov), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully.\n")

    # 统计容器
    snr_total_mod = defaultdict(int)          # 每个SNR的总样本数
    snr_correct_mod = defaultdict(int)        # 每个SNR正确分类数
    mod_total = {mod: defaultdict(int) for mod in range(12)}   # 每种调制每个SNR的总数
    mod_correct = {mod: defaultdict(int) for mod in range(12)} # 每种调制每个SNR的正确数
    
    all_pred_angles = []
    all_true_angles = []
    all_snrs = []

    with torch.no_grad():
        for batch in test_loader:
            # 由于 return_mod=True, return_fc=False，batch顺序：
            # cov, spectrum, angle, iq, mod, snr
            cov = batch[0].to(device)
            true_angle = batch[2].to(device)
            iq = batch[3].to(device)
            mod_label = batch[4].to(device)
            snr_batch = batch[5].cpu().numpy().flatten()   # 信噪比值

            # 前向推理
            mod_out, doa_out = model(iq, cov)

            # ----- 调制识别统计 -----
            pred_mod = torch.argmax(mod_out, dim=1).cpu().numpy()
            true_mod = mod_label.cpu().numpy()
            for i in range(len(pred_mod)):
                s = snr_batch[i]
                snr_total_mod[s] += 1
                if pred_mod[i] == true_mod[i]:
                    snr_correct_mod[s] += 1
                mod_total[true_mod[i]][s] += 1
                if pred_mod[i] == true_mod[i]:
                    mod_correct[true_mod[i]][s] += 1

            # ----- DOA 估计（使用模型输出的谱）-----
            doa_spectrum = torch.sigmoid(doa_out)       # 转换为谱概率
            doa_np = doa_spectrum.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(doa_np.shape[0]):
                angles, _ = findpeaks(doa_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                all_pred_angles.append(pred_angle)
                all_true_angles.append(true_angle_np[i])
                all_snrs.append(int(round(snr_batch[i])))

    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)

    # ========== 整体性能 ==========
    total_mod_correct = sum(snr_correct_mod.values())
    total_mod_samples = sum(snr_total_mod.values())
    overall_mod_acc = total_mod_correct / total_mod_samples * 100
    doa_errors = all_pred_angles - all_true_angles
    doa_rmse = np.sqrt(np.mean(doa_errors**2))
    doa_mae = np.mean(np.abs(doa_errors))

    print("\n========== Overall Performance (JointNet_LF) ==========")
    print(f"Modulation Accuracy: {overall_mod_acc:.2f}%")
    print(f"DOA RMSE: {doa_rmse:.4f}°, DOA MAE: {doa_mae:.4f}°\n")

    # ========== 按 SNR 分组性能 ==========
    unique_snrs = sorted(np.unique(all_snrs))
    print("========== SNR-wise Performance ==========")
    print(f"{'SNR (dB)':<10} {'Mod Acc(%)':<12} {'DOA RMSE(deg)':<15} {'DOA MAE(deg)':<15} {'Samples':<10}")
    print("-" * 65)
    for snr_val in unique_snrs:
        mod_acc = snr_correct_mod[snr_val] / snr_total_mod[snr_val] * 100 if snr_total_mod[snr_val] > 0 else 0
        mask = (all_snrs == snr_val)
        err = all_pred_angles[mask] - all_true_angles[mask]
        rmse = np.sqrt(np.mean(err**2)) if np.sum(mask) > 0 else 0
        mae = np.mean(np.abs(err)) if np.sum(mask) > 0 else 0
        n = np.sum(mask)
        print(f"{snr_val:<10} {mod_acc:<12.2f} {rmse:<15.4f} {mae:<15.4f} {n:<10}")

    # ========== 每种调制类型按 SNR 的准确率 ==========
    print("\n========== Per-modulation Accuracy by SNR ==========")
    for mod in range(12):
        if not mod_total[mod]:
            continue
        print(f"\nModulation {mod} ({mod_names[mod]}):")
        print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
        for snr_val in sorted(mod_total[mod].keys()):
            acc = mod_correct[mod][snr_val] / mod_total[mod][snr_val] * 100
            print(f"{snr_val:<10} {acc:<15.2f} {mod_total[mod][snr_val]:<10}")

    # ========== 可选：保存结果到文件 ==========
    results = {
        'overall': {'mod_acc': overall_mod_acc, 'doa_rmse': doa_rmse, 'doa_mae': doa_mae},
        'snr_wise': {
            s: {
                'mod_acc': snr_correct_mod[s]/snr_total_mod[s]*100 if snr_total_mod[s]>0 else 0,
                'doa_rmse': np.sqrt(np.mean((all_pred_angles[all_snrs==s] - all_true_angles[all_snrs==s])**2)) if np.sum(all_snrs==s)>0 else 0,
                'doa_mae': np.mean(np.abs(all_pred_angles[all_snrs==s] - all_true_angles[all_snrs==s])) if np.sum(all_snrs==s)>0 else 0,
                'samples': int(np.sum(all_snrs==s))
            } for s in unique_snrs
        },
        'per_mod_snr': {
            mod: {
                s: mod_correct[mod][s]/mod_total[mod][s]*100 if mod_total[mod][s]>0 else 0
                for s in mod_total[mod]
            } for mod in range(12) if mod_total[mod]
        }
    }
    with open(os.path.join(result_dir, 'joint_latent_test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    
def test_joint_latent_scatter():
    """
    测试指定 SNR 下的 DOA 估计性能，并保存 PDF 散点图。

    绘图说明：
    1. 横轴表示真实 DOA；
    2. 纵轴表示估计 DOA；
    3. 黑色虚线表示理想关系 theta_hat = theta；
    4. 红色空心菱形表示每个真实角度下的平均估计值；
    5. RMSE 根据该 SNR 下的全部原始测试样本计算；
    6. 不绘制标准差阴影，避免产生歧义。

    注意：
    H5MultiTaskDataset、JointNet_AdaptiveFusion 和 findpeaks
    需要已经在当前文件中定义，或从相应模块中导入。
    """
    test_data_path = (
        "/home/sp432sl/ZQ/DOA_regcition_0908/"
        "DIF_Net_0718/testdataset"
    )

    model_path = (
        "/home/sp432sl/ZQ/DOA_regcition_0908/"
        "DIF_Net_0718/model2/"
        "joint_latent_AdaptiveFusionjoint/"
        "best_model_joint.pth"
    )

    save_dir = (
        "/home/sp432sl/ZQ/DOA_regcition_0908/"
        "DIF_Net_0718/allaccrmse/scatter"
    )

    os.makedirs(save_dir, exist_ok=True)
    angle_min = -60
    angle_max = 60
    angle_step = 1

    DOA_grid = np.arange(
        angle_min,
        angle_max + angle_step,
        angle_step,
        dtype=np.float32,
    )

    num_doa_classes = len(DOA_grid)

    print("=" * 60)
    print("DOA estimation test")
    print(f"DOA range: {angle_min}° to {angle_max}°")
    print(f"DOA grid interval: {angle_step}°")
    print(f"Number of DOA grid points: {num_doa_classes}")
    print("=" * 60)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True,
        return_spectrum=True,
        return_angle=True,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        fc_norm="logminmax",
        iq_shape="2D",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"Number of test samples: {len(test_dataset)}")
    model = JointNet_AdaptiveFusion(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes,
        embed_dim=64,
        dropout_rate=0.3,
    ).to(device)
    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    load_result = model.load_state_dict(
        model_state_dict,
        strict=False,
    )

    if len(load_result.missing_keys) > 0:
        print("\nMissing keys:")
        for key in load_result.missing_keys:
            print(f"  {key}")

    if len(load_result.unexpected_keys) > 0:
        print("\nUnexpected keys:")
        for key in load_result.unexpected_keys:
            print(f"  {key}")

    model.eval()

    print("\nModel loaded successfully.")
    all_pred_angles = []
    all_true_angles = []
    all_snrs = []

    print("\nEvaluating test dataset...")

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            cov = batch[0].to(
                device,
                non_blocking=True,
            )

            true_angle = batch[2].to(
                device,
                non_blocking=True,
            )

            iq = batch[3].to(
                device,
                non_blocking=True,
            )

            snr_batch = (
                batch[5]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            _, doa_out = model(iq, cov)

            # 将输出转换为预测伪空间谱
            doa_spectrum = torch.sigmoid(doa_out)

            doa_np = (
                doa_spectrum
                .detach()
                .cpu()
                .numpy()
            )

            true_angle_np = (
                true_angle
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            for sample_idx in range(doa_np.shape[0]):

                estimated_angles, _ = findpeaks(
                    doa_np[sample_idx],
                    K=1,
                    DOA=DOA_grid,
                )

                estimated_angles = np.asarray(
                    estimated_angles
                ).reshape(-1)

                if estimated_angles.size == 0:
                    print(
                        "Warning: no peak was found for "
                        f"batch {batch_idx}, "
                        f"sample {sample_idx}."
                    )
                    continue

                pred_angle = float(
                    estimated_angles[0]
                )

                target_angle = float(
                    true_angle_np[sample_idx]
                )

                snr_value = int(
                    round(float(snr_batch[sample_idx]))
                )

                all_pred_angles.append(pred_angle)
                all_true_angles.append(target_angle)
                all_snrs.append(snr_value)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Processed {batch_idx + 1}/"
                    f"{len(test_loader)} batches"
                )

    all_pred_angles = np.asarray(
        all_pred_angles,
        dtype=np.float32,
    )

    all_true_angles = np.asarray(
        all_true_angles,
        dtype=np.float32,
    )

    all_snrs = np.asarray(
        all_snrs,
        dtype=np.int32,
    )

    print(
        f"\nTotal successfully evaluated samples: "
        f"{len(all_pred_angles)}"
    )

    if len(all_pred_angles) == 0:
        print("No valid prediction results were obtained.")
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    snr_list = [-8, -6, 12]

    for snr_val in snr_list:
        snr_mask = (all_snrs == snr_val)

        pred_snr = all_pred_angles[snr_mask]
        true_snr = all_true_angles[snr_mask]

        print(
            f"\nSNR = {snr_val} dB, "
            f"number of samples = {len(pred_snr)}"
        )

        if len(pred_snr) == 0:
            print(
                f"No samples were found at "
                f"SNR = {snr_val} dB. Skip."
            )
            continue
        unique_angles = np.sort(
            np.unique(true_snr)
        )

        mean_pred_angles = []
        valid_true_angles = []
        for angle in unique_angles:

            angle_mask = np.isclose(
                true_snr,
                angle,
                atol=1e-6,
            )

            preds_at_angle = pred_snr[angle_mask]

            if len(preds_at_angle) == 0:
                continue

            mean_pred_angle = np.mean(
                preds_at_angle
            )

            valid_true_angles.append(angle)
            mean_pred_angles.append(
                mean_pred_angle
            )

        valid_true_angles = np.asarray(
            valid_true_angles,
            dtype=np.float32,
        )

        mean_pred_angles = np.asarray(
            mean_pred_angles,
            dtype=np.float32,
        )
        errors = pred_snr - true_snr

        overall_rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        overall_mae = np.mean(
            np.abs(errors)
        )

        print(f"Overall RMSE: {overall_rmse:.4f}°")
        print(f"Overall MAE:  {overall_mae:.4f}°")
        fig, ax = plt.subplots(
            figsize=(6, 6)
        )
        true_line, = ax.plot(
            [angle_min, angle_max],
            [angle_min, angle_max],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="True DOA",
            zorder=1,
        )
        estimated_points = ax.scatter(
            valid_true_angles,
            mean_pred_angles,
            alpha=0.7,
            marker="D",
            edgecolors="red",
            facecolors="none",
            s=30,
            linewidths=0.5,
            label="Estimated DOA",
            zorder=2,
        )
        ax.set_xlabel("True DOA (°)")
        ax.set_ylabel("Estimated DOA (°)")

        ax.set_xlim(
            angle_min,
            angle_max,
        )

        ax.set_ylim(
            angle_min,
            angle_max,
        )

        ax.set_xticks(
            np.arange(-60, 61, 20)
        )

        ax.set_yticks(
            np.arange(-60, 61, 20)
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6,
        )
        ax.legend(
            handles=[
                true_line,
                estimated_points,
            ],
            labels=[
                "True DOA",
                "Estimated DOA",
            ],
            loc="upper left",
            frameon=True,
            fancybox=False,
        )
        ax.text(
            0.96,
            0.05,
            f"RMSE = {overall_rmse:.2f}°",
            transform=ax.transAxes,
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="0.5",
                alpha=1.0,
            ),
        )
        fig.tight_layout()
        save_path = os.path.join(
            save_dir,
            f"DOA_scatter_SNR_{snr_val}dB.pdf",
        )

        fig.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Scatter plot saved to: {save_path}"
        )

    print(
        "\nAll DOA scatter plots generated successfully."
    )


# 不同长度
def repeat_pad_to_length(iq, target_len=768):
    """
    将 IQ 信号填充/截断到目标长度
    iq: (B, 2, L)  或 (B, 2, L, 1)  或 (B, 2, L)
    返回: (B, 2, target_len)  保持2维通道
    """
    # 统一去掉多余的维度
    if iq.dim() == 4:
        iq = iq.squeeze(-1)   # (B,2,L)
    if iq.dim() == 2:
        iq = iq.unsqueeze(0)  # (1,2,L)
    B, C, L = iq.shape
    if L == target_len:
        return iq
    elif L < target_len:
        repeat_times = (target_len + L - 1) // L
        iq = iq.repeat(1, 1, repeat_times)
        return iq[:, :, :target_len]
    else:
        return iq[:, :, :target_len]

def sliding_window_inference_joint(iq, cov, model, window_size=768, stride=256):
    """
    对长序列 IQ 进行滑动窗口推理，融合调制和 DOA 结果
    iq:  (B, 2, L)  原始长序列
    cov: (B, 2, M, M)  协方差矩阵（所有窗口共用）
    model: JointNet_LF 实例
    返回:
        mod_out_avg: (B, num_classes_mod)  平均 logits
        doa_out_avg: (B, num_classes_doa)  平均 logits
    """
    B, C, L = iq.shape
    # 确保 iq 是 (B,2,L) 形状
    if iq.dim() == 4:
        iq = iq.squeeze(-1)

    mod_logits_list = []
    doa_logits_list = []

    for start in range(0, L - window_size + 1, stride):
        iq_win = iq[:, :, start:start+window_size]          # (B,2,window_size)
        iq_win = iq_win.unsqueeze(-1)                       # (B,2,window_size,1)
        mod_out, doa_out = model(iq_win, cov)
        mod_logits_list.append(mod_out)   # (B, num_mod)
        doa_logits_list.append(doa_out)   # (B, num_doa)

    # 沿窗口维度平均
    mod_out_avg = torch.stack(mod_logits_list, dim=0).mean(dim=0)   # (B, num_mod)
    doa_out_avg = torch.stack(doa_logits_list, dim=0).mean(dim=0)   # (B, num_doa)
    return mod_out_avg, doa_out_avg

def test_joint_latent_full_generalized():
    """
    测试 JointNet_LF 模型，支持任意长度 IQ 输入（自动填充/滑动窗口）
    """
    # 测试数据路径（可修改）
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/testdataset'
    test_data_path ='/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/diffM/M10'
    #M=8"
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/datasetdifffsnap/-20.25/256'
    # model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/model2/joint_latent_AdaptiveFusionjoint/best_model_joint.pth'

    #M=7
    # test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/datasetdifffsnap/5.1/4482'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/model2/joint_latent_AdaptiveFusionjoint/best_model_joint.pth'
    # os.makedirs(result_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DOA 角度网格
    angle_min, angle_max, step = -60, 60, 1
    DOA_grid = np.linspace(angle_min, angle_max, int((angle_max - angle_min)/step) + 1)
    num_doa_classes = len(DOA_grid)

    # 调制类型名称
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    # 加载测试数据集（返回 IQ 和协方差）
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True, return_spectrum=True, return_angle=True,
        return_iq=True, return_mod=True, return_snr=True,
        return_fc=False,           # 隐式频率模型不需要外部fc标签
        fc_norm='logminmax', iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 加载模型
    model = JointNet_AdaptiveFusion(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes,
        embed_dim=64,
        dropout_rate=0.3
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully.\n")

    # 统计容器
    snr_total_mod = defaultdict(int)
    snr_correct_mod = defaultdict(int)
    mod_total = {mod: defaultdict(int) for mod in range(12)}
    mod_correct = {mod: defaultdict(int) for mod in range(12)}

    all_pred_angles = []
    all_true_angles = []
    all_snrs = []

    with torch.no_grad():
        for batch in test_loader:
            # 数据顺序: cov, spectrum, angle, iq, mod, snr
            cov = batch[0].to(device)
            true_angle = batch[2].to(device)
            iq = batch[3]               # (B,2,L) 或 (B,2,L,1)
            mod_label = batch[4].to(device)
            snr_batch = batch[5].cpu().numpy().flatten()

            # 处理 IQ 长度 (变长支持)
            L = iq.shape[-1] if iq.dim() == 3 else iq.shape[-2]  # 获取时间长度
            if L < 768:
                iq_proc = repeat_pad_to_length(iq, 768)
                iq_proc = iq_proc.unsqueeze(-1).to(device)  # (B,2,768,1)
                mod_out, doa_out = model(iq_proc, cov)
            elif L == 768:
                if iq.dim() == 3:
                    iq = iq.unsqueeze(-1)
                iq = iq.to(device)
                mod_out, doa_out = model(iq, cov)
            else:
                # 长序列：滑动窗口推理
                iq = iq.to(device)
                if iq.dim() == 4:
                    iq = iq.squeeze(-1)
                mod_out, doa_out = sliding_window_inference_joint(iq, cov, model, window_size=768, stride=256)

            # ----- 调制识别统计 -----
            pred_mod = torch.argmax(mod_out, dim=1).cpu().numpy()
            true_mod = mod_label.cpu().numpy()
            for i in range(len(pred_mod)):
                s = snr_batch[i]
                snr_total_mod[s] += 1
                if pred_mod[i] == true_mod[i]:
                    snr_correct_mod[s] += 1
                mod_total[true_mod[i]][s] += 1
                if pred_mod[i] == true_mod[i]:
                    mod_correct[true_mod[i]][s] += 1

            # ----- DOA 估计 -----
            doa_spectrum = torch.sigmoid(doa_out)       # 转换为谱概率
            doa_np = doa_spectrum.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(doa_np.shape[0]):
                angles, _ = findpeaks(doa_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                all_pred_angles.append(pred_angle)
                all_true_angles.append(true_angle_np[i])
                all_snrs.append(int(round(snr_batch[i])))

    # 转换为 numpy 数组
    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)

    # ========== 整体性能 ==========
    total_mod_correct = sum(snr_correct_mod.values())
    total_mod_samples = sum(snr_total_mod.values())
    overall_mod_acc = total_mod_correct / total_mod_samples * 100
    doa_errors = all_pred_angles - all_true_angles
    doa_rmse = np.sqrt(np.mean(doa_errors**2))
    doa_mae = np.mean(np.abs(doa_errors))

    print("\n========== Overall Performance (JointNet_LF with variable length) ==========")
    print(f"Modulation Accuracy: {overall_mod_acc:.2f}%")
    print(f"DOA RMSE: {doa_rmse:.4f}°, DOA MAE: {doa_mae:.4f}°\n")

    # ========== 按 SNR 分组 ==========
    unique_snrs = sorted(np.unique(all_snrs))
    print("========== SNR-wise Performance ==========")
    print(f"{'SNR (dB)':<10} {'Mod Acc(%)':<12} {'DOA RMSE(deg)':<15} {'DOA MAE(deg)':<15} {'Samples':<10}")
    print("-" * 65)
    for snr_val in unique_snrs:
        mod_acc = snr_correct_mod[snr_val] / snr_total_mod[snr_val] * 100 if snr_total_mod[snr_val] > 0 else 0
        mask = (all_snrs == snr_val)
        err = all_pred_angles[mask] - all_true_angles[mask]
        rmse = np.sqrt(np.mean(err**2)) if np.sum(mask) > 0 else 0
        mae = np.mean(np.abs(err)) if np.sum(mask) > 0 else 0
        n = np.sum(mask)
        print(f"{snr_val:<10} {mod_acc:<12.2f} {rmse:<15.4f} {mae:<15.4f} {n:<10}")

    # ========== 每种调制类型按 SNR 的准确率 ==========
    print("\n========== Per-modulation Accuracy by SNR ==========")
    for mod in range(12):
        if not mod_total[mod]:
            continue
        print(f"\nModulation {mod} ({mod_names[mod]}):")
        print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
        for snr_val in sorted(mod_total[mod].keys()):
            acc = mod_correct[mod][snr_val] / mod_total[mod][snr_val] * 100
            print(f"{snr_val:<10} {acc:<15.2f} {mod_total[mod][snr_val]:<10}")

def test_joint_latent_full_allresult():
    """
    测试 JointNet_LF 模型，支持任意长度 IQ 输入（自动填充/滑动窗口）
    输出按调制类型+SNR的调制准确率和DOA RMSE/MAE，并保存到文件
    """
    # 测试数据路径（可修改）
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/testdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/model2/joint_latent_AdaptiveFusionjoint/best_model_joint.pth'
    save_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/allaccrmse'
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'per_mod_snr_performance.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DOA 角度网格
    angle_min, angle_max, step = -60, 60, 1
    DOA_grid = np.linspace(angle_min, angle_max, int((angle_max - angle_min)/step) + 1)
    num_doa_classes = len(DOA_grid)

    # 调制类型名称
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    # 加载测试数据集
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True, return_spectrum=True, return_angle=True,
        return_iq=True, return_mod=True, return_snr=True,
        return_fc=False, fc_norm='logminmax', iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 加载模型
    model = JointNet_AdaptiveFusion(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes,
        embed_dim=64,
        dropout_rate=0.3
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully.\n")

    # ---------- 统计容器 ----------
    # 调制识别统计
    snr_total_mod = defaultdict(int)
    snr_correct_mod = defaultdict(int)
    mod_total = {mod: defaultdict(int) for mod in range(12)}
    mod_correct = {mod: defaultdict(int) for mod in range(12)}

    # DOA 误差统计
    all_pred_angles = []
    all_true_angles = []
    all_snrs = []
    all_true_mods = []

    # 按调制类型+SNR统计误差平方和、绝对误差和、样本数
    mod_doa_stats = {mod: defaultdict(lambda: {'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0}) for mod in range(12)}

    with torch.no_grad():
        for batch in test_loader:
            cov = batch[0].to(device)
            true_angle = batch[2].to(device)
            iq = batch[3]
            mod_label = batch[4].to(device)
            snr_batch = batch[5].cpu().numpy().flatten()

            # 处理 IQ 长度 (变长支持)
            L = iq.shape[-1] if iq.dim() == 3 else iq.shape[-2]
            if L < 768:
                iq_proc = repeat_pad_to_length(iq, 768)
                iq_proc = iq_proc.unsqueeze(-1).to(device)
                mod_out, doa_out = model(iq_proc, cov)
            elif L == 768:
                if iq.dim() == 3:
                    iq = iq.unsqueeze(-1)
                iq = iq.to(device)
                mod_out, doa_out = model(iq, cov)
            else:
                iq = iq.to(device)
                if iq.dim() == 4:
                    iq = iq.squeeze(-1)
                mod_out, doa_out = sliding_window_inference_joint(iq, cov, model, window_size=768, stride=256)

            # 调制识别统计
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

            # DOA 估计
            doa_spectrum = torch.sigmoid(doa_out)
            doa_np = doa_spectrum.cpu().numpy()
            true_angle_np = true_angle.cpu().numpy()
            for i in range(doa_np.shape[0]):
                angles, _ = findpeaks(doa_np[i], K=1, DOA=DOA_grid)
                pred_angle = angles[0]
                err = pred_angle - true_angle_np[i]
                all_pred_angles.append(pred_angle)
                all_true_angles.append(true_angle_np[i])
                all_snrs.append(int(round(snr_batch[i])))
                all_true_mods.append(true_mod[i])

                mod = true_mod[i]
                snr = int(round(snr_batch[i]))
                mod_doa_stats[mod][snr]['sum_sq'] += err ** 2
                mod_doa_stats[mod][snr]['sum_abs'] += abs(err)
                mod_doa_stats[mod][snr]['count'] += 1

    # 转换为 numpy 数组
    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)
    all_snrs = np.array(all_snrs)
    all_true_mods = np.array(all_true_mods)

    # ========== 整体性能（终端输出） ==========
    total_mod_correct = sum(snr_correct_mod.values())
    total_mod_samples = sum(snr_total_mod.values())
    overall_mod_acc = total_mod_correct / total_mod_samples * 100
    doa_errors = all_pred_angles - all_true_angles
    doa_rmse = np.sqrt(np.mean(doa_errors**2))
    doa_mae = np.mean(np.abs(doa_errors))

    print("\n========== Overall Performance (JointNet_LF with variable length) ==========")
    print(f"Modulation Accuracy: {overall_mod_acc:.2f}%")
    print(f"DOA RMSE: {doa_rmse:.4f}°, DOA MAE: {doa_mae:.4f}°\n")

    # ========== 按 SNR 分组 ==========
    unique_snrs = sorted(np.unique(all_snrs))
    print("========== SNR-wise Performance ==========")
    print(f"{'SNR (dB)':<10} {'Mod Acc(%)':<12} {'DOA RMSE(deg)':<15} {'DOA MAE(deg)':<15} {'Samples':<10}")
    print("-" * 65)
    for snr_val in unique_snrs:
        mod_acc = snr_correct_mod[snr_val] / snr_total_mod[snr_val] * 100 if snr_total_mod[snr_val] > 0 else 0
        mask = (all_snrs == snr_val)
        err = all_pred_angles[mask] - all_true_angles[mask]
        rmse = np.sqrt(np.mean(err**2)) if np.sum(mask) > 0 else 0
        mae = np.mean(np.abs(err)) if np.sum(mask) > 0 else 0
        n = np.sum(mask)
        print(f"{snr_val:<10} {mod_acc:<12.2f} {rmse:<15.4f} {mae:<15.4f} {n:<10}")

    # ========== 准备保存的内容（所有表格字符串） ==========
    lines = []
    lines.append("========== Per-modulation Accuracy by SNR ==========")
    for mod in range(12):
        if not mod_total[mod]:
            continue
        lines.append(f"\nModulation {mod} ({mod_names[mod]}):")
        lines.append(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'Samples':<10}")
        for snr_val in sorted(mod_total[mod].keys()):
            acc = mod_correct[mod][snr_val] / mod_total[mod][snr_val] * 100
            lines.append(f"{snr_val:<10} {acc:<15.2f} {mod_total[mod][snr_val]:<10}")

    lines.append("\n========== Per-modulation DOA RMSE/MAE by SNR ==========")
    for mod in range(12):
        if not mod_doa_stats[mod]:
            continue
        lines.append(f"\nModulation {mod} ({mod_names[mod]}):")
        lines.append(f"{'SNR (dB)':<10} {'RMSE (deg)':<15} {'MAE (deg)':<15} {'Samples':<10}")
        for snr_val in sorted(mod_doa_stats[mod].keys()):
            stats = mod_doa_stats[mod][snr_val]
            count = stats['count']
            if count == 0:
                continue
            rmse = np.sqrt(stats['sum_sq'] / count)
            mae = stats['sum_abs'] / count
            lines.append(f"{snr_val:<10} {rmse:<15.4f} {mae:<15.4f} {count:<10}")

    # ========== 写入文件 ==========
    with open(save_file, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nPer-modulation performance saved to {save_file}")

def generate_confusion_matrices(snr_list=[-8, -4, 0]):
    """
    绘制并保存指定 SNR 下的调制识别混淆矩阵（PDF 格式，高分辨率）

    Args:
        snr_list: list of int/float，需要绘制的 SNR 值，默认 [-8, -4, 0]
    """
    # ---------- 配置路径 ----------
    test_data_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/testdataset'
    model_path = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/model2/joint_latent_AdaptiveFusionjoint/best_model_joint.pth'
    result_dir = '/home/sp432sl/ZQ/DOA_regcition_0908/DIF_Net_0718/allaccrmse/confusion'
    os.makedirs(result_dir, exist_ok=True)   # 确保目录存在

    # ---------- 设备 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- 数据集 ----------
    from tool import H5MultiTaskDataset   # 请确保 tool.py 中定义了该类
    test_dataset = H5MultiTaskDataset(
        folder_path=test_data_path,
        return_cov=True,
        return_spectrum=True,
        return_angle=True,
        return_iq=True,
        return_mod=True,
        return_snr=True,
        return_fc=False,
        fc_norm='logminmax',
        iq_shape='2D'
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # ---------- 模型 ----------
    angle_min, angle_max, step = -60, 60, 1
    DOA_grid = np.linspace(angle_min, angle_max, int((angle_max - angle_min)/step) + 1)
    num_doa_classes = len(DOA_grid)

    model = JointNet_AdaptiveFusion(
        num_classes_mod=12,
        num_classes_doa=num_doa_classes,
        embed_dim=64,
        dropout_rate=0.3
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully.\n")

    # ---------- 收集指定 SNR 下的预测和真实标签 ----------
    snr_to_labels = {snr: {'true': [], 'pred': []} for snr in snr_list}
    mod_names = ["FSK4", "BPSK", "LFM", "FRANK", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

    with torch.no_grad():
        for batch in test_loader:
            # 根据 H5MultiTaskDataset 返回值的顺序：
            # cov, spectrum, angle, iq, mod, snr
            cov = batch[0].to(device)
            iq = batch[3].to(device)
            mod_label = batch[4].to(device)
            snr_batch = batch[5].cpu().numpy().flatten()

            # 前向推理（只需要调制输出）
            mod_out, _ = model(iq, cov)   # _ 是 DOA 输出，忽略
            pred_mod = torch.argmax(mod_out, dim=1).cpu().numpy()
            true_mod = mod_label.cpu().numpy()

            for i in range(len(pred_mod)):
                s = int(round(snr_batch[i]))   # 转为整数便于匹配
                if s in snr_to_labels:
                    snr_to_labels[s]['true'].append(true_mod[i])
                    snr_to_labels[s]['pred'].append(pred_mod[i])

    # ---------- 绘制混淆矩阵（保存为 PDF） ----------
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
        ax = sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                         xticklabels=mod_names, yticklabels=mod_names,
                         cbar_kws={'label': 'Accuracy (%)'})
        plt.xlabel('Predicted Modulation')
        plt.ylabel('True Modulation')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存为 PDF（矢量格式，清晰度高）
        save_path = os.path.join(result_dir, f'confusion_matrix_SNR_{snr_val}dB.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    print("\nAll confusion matrices generated (PDF format).")

# ----------------------- 主程序 ----------------------- #
if __name__ == "__main__":
    #===== 主训练

    # main_jointnet_latent(mode='joint')

    # test_joint_latent_full()#隐式
    # test_joint_latent_full_allresult()
    # test_joint_latent_full_generalized()

    #======DOA散点图
    test_joint_latent_scatter()

    #======调制识别混淆矩阵
#    generate_confusion_matrices(snr_list=[-8, -4, 8])


