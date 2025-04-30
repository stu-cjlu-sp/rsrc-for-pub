from utils import get_dataloader_2018a_IQ
from model_v0 import SEFEFeatureExtractor
import os, time, torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

# 设置参数
# Settings
# MODEL = "SEFE_PeriodAwareMamba_MambaLocal"
# MODEL = "SEFEMambaWithPeriod"
# MODEL = "SEFE_PeriodicConv_MambaLocal"
MODEL = "MambaSEFEMamba"
# MODEL = "SEFEMambaLocal"
# MODEL = "SEFEMamba"
# MODEL = "SEFE"

DATASET = "Data-2018a_IQ"
LOG_FOLDER = f"logs/{DATASET}_{MODEL}/test"
os.makedirs(LOG_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型构建（与2016版相同）
model_dict = {
    # "CNN": CNNModel,
    # "CNNLSTM": CNNLSTMModel,
    "SEFE": SEFEFeatureExtractor
    # "SEFEMamba": SEFEMambaFeatureExtractor,
    # "SEFEMambaLocal": SEFEMambaLocal,
    # "SEFE_PeriodicConv_MambaLocal": SEFE_PeriodicConv_MambaLocal,
    # "SEFEMambaWithPeriod": SEFEMambaWithPeriod,
    # "SEFE_PeriodAwareMamba_MambaLocal": SEFE_PeriodAwareMamba_MambaLocal,
    # "MambaSEFEMamba": MambaSEFEMamba
}
model = model_dict[MODEL](input_dim=2, seq_len=1024, num_classes=24).to(device)

# 加载模型
NAME = f"Model[{MODEL}]_Data-[{DATASET}]"
resume_path = f"logs/{DATASET}_{MODEL}/net_{NAME}.pth"
print(f"[Test] Checkpoint = {resume_path}")
checkpoint = torch.load(resume_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

# 加载数据
_, _, test_loader = get_dataloader_2018a_IQ(batch_size=256)

# SNR 标签复原表
snr_mapping = []
for mod in range(24):
    for snr_idx in range(26):
        snr_val = -20 + 2 * snr_idx
        snr_mapping.extend([snr_val] * 4096)
snr_mapping = np.array(snr_mapping)

# 推理
model.eval()
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
        all_snrs.extend(snrs.cpu().numpy())
        total_batches += 1
        total_samples += inputs.size(0)

# === 准确率统计 ===
acc_per_snr = defaultdict(list)
for true, pred, snr in zip(all_labels, all_preds, all_snrs):
    acc_per_snr[snr].append(int(true == pred))

snrs_sorted = sorted(acc_per_snr.keys())
accuracy_per_snr = [np.mean(acc_per_snr[snr]) for snr in snrs_sorted]

# === 保存准确率图像 ===
plt.figure(figsize=(8, 5))
plt.plot(snrs_sorted, accuracy_per_snr, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs SNR (2018a)")
plt.grid(True)
plt.savefig(os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Accuracy_vs_SNR.png"))
plt.close()

# === 混淆矩阵 per SNR ===
label_names = [  # 和 mod_classes 一致
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
    'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]
label_indics = list(range(24))

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
        text.set_fontsize(5)
    plt.title(f"Confusion Matrix @ SNR = {snr} dB")
    plt.tight_layout()
    save_path = os.path.join(LOG_FOLDER, f"{MODEL}_{DATASET}_Confusion_Matrix_SNR_{snr}dB.png")
    plt.savefig(save_path)
    plt.close()

# === 推理时间 ===
print(f"[✓] Avg inference time per batch: {1000 * total_time / total_batches:.2f} ms")
print(f"[✓] Avg inference time per sample: {1000 * total_time / total_samples:.4f} ms")


# === 准确率统计结果 ===
max_acc = max(accuracy_per_snr)
max_snr = snrs_sorted[accuracy_per_snr.index(max_acc)]
avg_acc_low_snr = np.mean([np.mean(acc_per_snr[snr]) for snr in snrs_sorted if -20 <= snr <= 0])
avg_acc_high_snr = np.mean([np.mean(acc_per_snr[snr]) for snr in snrs_sorted if 0 < snr <= 30])
overall_acc = np.mean([true == pred for true, pred in zip(all_labels, all_preds)])

print(f"[✓] Highest Accuracy: {max_acc:.4f} @ SNR={max_snr} dB")
print(f"[✓] Avg Accuracy [-20~0 dB]: {avg_acc_low_snr:.4f}")
print(f"[✓] Avg Accuracy [0~30 dB]: {avg_acc_high_snr:.4f}")
print(f"[✓] Overall Accuracy: {overall_acc:.4f}")

# === 写入文本 ===
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

