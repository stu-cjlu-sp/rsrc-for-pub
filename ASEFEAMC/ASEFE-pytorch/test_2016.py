import re
import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
from model_v0 import SEFEFeatureExtractor

from data import Radioml_16_Dataset

# Settings
# MODEL = "SEFE_PeriodAwareMamba_MambaLocal"
# MODEL = "SEFEMambaWithPeriod"
# MODEL = "SEFE_PeriodicConv_MambaLocal"
# MODEL = "MambaSEFEMamba"
MODEL = "SEFEMambaLocal"
# MODEL = "SEFEMamba"
# MODEL = "SEFE"
# MODEL = "CNNLSTM"
# MODEL = "CNN"
DATASET = "Data-2016a_IQ"
LOG_FOLDER = f"logs/{DATASET}_{MODEL}/test"
os.makedirs(LOG_FOLDER, exist_ok=True)

SNR_list = list(range(-20, 20, 2))  # full SNR range
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
# if MODEL == "CNN":
#     model = CNNModel().to(device)
# elif MODEL == "CNNLSTM":
#     model = CNNLSTMModel().to(device)
if MODEL == "SEFE":
    model = SEFEFeatureExtractor(input_dim=2, seq_len=128, num_classes=11).to(device)
# elif MODEL == "SEFEMamba":
#     model = SEFEMambaFeatureExtractor(input_dim=2, seq_len=128, num_classes=11).to(device)
# elif MODEL == "SEFEMambaLocal":
#     model = SEFEMambaLocal(input_dim=2, seq_len=128, num_classes=11).to(device)
# elif MODEL == "SEFE_PeriodicConv_MambaLocal":
#     model = SEFE_PeriodicConv_MambaLocal(input_dim=2, seq_len=128, num_classes=11).to(device)
# elif MODEL == "SEFEMambaWithPeriod":
#     # (Mamba, LearnablePeriod Estimator, SEFE, PeriodicConv, Mamba, LocalEnhancer)
#     model = SEFEMambaWithPeriod(input_dim=2, seq_len=128, num_classes=11).to(device)
# elif MODEL == "SEFE_PeriodAwareMamba_MambaLocal":
#     model = SEFE_PeriodAwareMamba_MambaLocal(input_dim=2, seq_len=128, num_classes=11).to(device)
# # 对比实验
# elif MODEL == "MambaSEFEMamba":
#     model = MambaSEFEMamba(input_dim=2, seq_len=128, num_classes=11).to(device)
else:
    raise ValueError("Please choose a valid model")

# ------------------ 模型参数量统计 -------------------
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
# ----------------------------------------------------

# Load test data
dataset2016 = Radioml_16_Dataset("/home/song/Datasets/RML2016.10a_dict.pkl")
test_loader = dataset2016.test_loader

# Restore model
NAME = f"Model[{MODEL}]_Data-[{DATASET}]"
resume_path = f"logs/{DATASET}_{MODEL}/net_{NAME}.pth"

print(f"[Test] Checkpoint = {resume_path}")
if not os.path.isfile(resume_path):
    print(f"  [!] Checkpoint not found.")

checkpoint = torch.load(resume_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

# Inference
all_labels, all_preds, all_snrs = [], [], []
total_time, total_batches, total_samples = 0, 0, 0

with torch.no_grad():
    for inputs, labels, snrs in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()
        outputs = model(inputs)
        total_time += time.time() - start

        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_snrs.extend(snr.item() for snr in snrs.cpu().numpy())
        total_batches += 1
        total_samples += inputs.size(0)

# Accuracy by SNR
acc_per_snr = defaultdict(list)
for true, pred, snr in zip(all_labels, all_preds, all_snrs):
    acc_per_snr[snr].append(int(true == pred))

snrs_sorted = sorted(acc_per_snr.keys())
accuracy_per_snr = [np.mean(acc_per_snr[snr]) for snr in snrs_sorted]

# Save Accuracy vs SNR Plot
fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
ax_acc.plot(snrs_sorted, accuracy_per_snr, marker='o')
ax_acc.set_xlabel("SNR (dB)")
ax_acc.set_ylabel("Test Accuracy")
ax_acc.set_title("Test Accuracy vs SNR")
ax_acc.grid(True)
ax_acc.set_ylim(0, 1.0)
ax_acc.set_yticks(np.arange(0, 1.01, 0.1))
ax_acc.set_xticks(SNR_list)
plt.savefig(os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Accuracy_vs_SNR.png"))
plt.close()

# Overall Confusion Matrix
# cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(10, 20))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# plt.title("Overall Confusion Matrix")
# plt.savefig(os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Confusion_Matrix.png"))
# plt.close()

# 标签类别（确保排序一致）
label_indics = sorted(list(set(all_labels)))
label_names = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
# 每个 SNR 下的混淆矩阵
print("[✓] Drawing per-SNR confusion matrices...")
for snr in snrs_sorted:
    indices = [i for i, s in enumerate(all_snrs) if s == snr]
    labels_snr = [all_labels[i] for i in indices]
    preds_snr = [all_preds[i] for i in indices]

    cm_snr = confusion_matrix(labels_snr, preds_snr, labels=label_indics)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    for text in disp.text_.ravel():
        text.set_fontsize(6)
    plt.title(f"Confusion Matrix @ SNR = {snr} dB")
    plt.tight_layout()
    save_path = os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Confusion_Matrix_SNR_{snr}dB.png")
    plt.savefig(save_path)
    plt.close()

# Summary timing
print(f"[✓] Avg inference time per batch: {1000 * total_time / total_batches:.2f} ms")
print(f"[✓] Avg inference time per sample: {1000 * total_time / total_samples:.4f} ms")

# --- Statistics ---
# Highest Accuracy
max_acc = max(accuracy_per_snr)
max_snr = snrs_sorted[accuracy_per_snr.index(max_acc)]

# SNR 区间划分统计
acc_low_snr = [acc_per_snr[snr] for snr in snrs_sorted if -20 <= snr <= 0]
acc_high_snr = [acc_per_snr[snr] for snr in snrs_sorted if 0 < snr <= 18]

avg_acc_low_snr = np.mean([np.mean(x) for x in acc_low_snr]) if acc_low_snr else 0
avg_acc_high_snr = np.mean([np.mean(x) for x in acc_high_snr]) if acc_high_snr else 0

# Overall Accuracy
overall_acc = np.mean([true == pred for true, pred in zip(all_labels, all_preds)])

print(f"[✓] Highest Accuracy: {max_acc:.4f} @ SNR={max_snr} dB")
print(f"[✓] Avg Accuracy [-20~0 dB]: {avg_acc_low_snr:.4f}")
print(f"[✓] Avg Accuracy [0~18 dB]: {avg_acc_high_snr:.4f}")
print(f"[✓] Overall Accuracy: {overall_acc:.4f}")

# Save TXT
txt_path = os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Accuracy_vs_SNR.txt")
with open(txt_path, 'w') as f:
    f.write("SNR\tAccuracy\n")
    for snr, acc in zip(snrs_sorted, accuracy_per_snr):
        f.write(f"{snr}\t{acc:.6f}\n")
    f.write("\n")
    f.write(f"Highest Accuracy: {max_acc:.6f} @ SNR={max_snr} dB\n")
    f.write(f"Avg Accuracy [-20~0 dB]: {avg_acc_low_snr:.6f}\n")
    f.write(f"Avg Accuracy [0~18 dB]: {avg_acc_high_snr:.6f}\n")
    f.write(f"Overall Accuracy: {overall_acc:.6f}\n\n")

    f.write(f"[✓] Avg inference time per batch: {1000 * total_time / total_batches:.2f} ms\n")
    f.write(f"[✓] Avg inference time per sample: {1000 * total_time / total_samples:.4f} ms\n")

# 设置路径
LOG_FOLDER = "logs/{}_{}".format(DATASET, MODEL)
log_path = os.path.join(LOG_FOLDER, 'training_%s.log' % NAME)

# 匹配 Step Loss 的正则表达式
step_loss_pattern = re.compile(r"Epoch \[(\d+)\] Step \[(\d+)\] Loss: ([0-9.]+)")

# 解析日志
epoch_losses = {}

with open(log_path, 'r') as f:
    for line in f:
        match = step_loss_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(3))
            epoch_losses.setdefault(epoch, []).append(loss)

# 计算均值与标准差
epochs = sorted(epoch_losses.keys())
mean_losses = [np.mean(epoch_losses[e]) for e in epochs]
std_losses = [np.std(epoch_losses[e]) for e in epochs]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_losses, label="Train Loss", color='blue')
plt.fill_between(epochs,
                 np.array(mean_losses) - np.array(std_losses),
                 np.array(mean_losses) + np.array(std_losses),
                 alpha=0.3, color='blue', label='Std Dev')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch with Std Dev")
plt.legend()
plt.grid(True)

# 保存图像
save_path = os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Train_Loss_Epoch_Std.png")
plt.savefig(save_path)
plt.close()

print(f"[✓] Training Loss curve with Std Dev saved to {save_path}")