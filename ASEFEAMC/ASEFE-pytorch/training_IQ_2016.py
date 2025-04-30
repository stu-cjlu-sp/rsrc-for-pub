import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchinfo import summary
from model_v0 import CNNModel, CNNLSTMModel, SEFEFeatureExtractor, SEFEMambaFeatureExtractor, SEFEMambaLocal, \
    MambaSEFEMamba, SEFE_PeriodicConv_MambaLocal, SEFEMambaWithPeriod, SEFE_PeriodAwareMamba_MambaLocal
from utils import datalodaer_2016a_IQ, get_dataframe, get_dataloader_2018a_IQ

# Settings
# MODEL = "SEFE_PeriodAwareMamba_MambaLocal"
# MODEL = "SEFEMambaWithPeriod"
# MODEL = "SEFE_PeriodicConv_MambaLocal"
# MODEL = "MambaSEFEMamba"
# MODEL = "SEFEMambaLocal"
MODEL = "SEFEMamba"
# MODEL = "SEFE"
# MODEL = "CNNLSTM" # only for 2016 dataset
# MODEL = "CNN" # only for 2016 dataset
DATASET = "Data-2016a_IQ" #"Data-2018a_IQ" #
LOG_FOLDER = "logs/{}_{}".format(DATASET, MODEL)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
num_epochs = 100
milestone = 30 # for learning rate
lr = 0.001
# SNR_list = [n for n in range(-20, 20, 2)]  # list of SNRs: [-20,-18....,16,18]  2016a dataset
# SNR_list = [6] # debug
if DATASET == "Data-2016a_IQ":
    num_classes = 11
    seq_len = 128
elif DATASET == "Data-2018a_IQ":
    num_classes = 24
    seq_len = 1024
    # SNR_list = [n for n in range(-20, 32, 2)]  # list of SNRs: [-20,-18....,26,28,30]  2016a dataset
    num_epochs = 50
else:
    num_classes = 11 # default
    seq_len = 128

# 设置模型运行的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
if MODEL == "CNN":
    model = CNNModel().to(device)
elif MODEL == "CNNLSTM":
    model = CNNLSTMModel().to(device)
elif MODEL == "SEFE":
    model = SEFEFeatureExtractor(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)
elif MODEL == "SEFEMamba":
    model = SEFEMambaFeatureExtractor(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)
elif MODEL == "SEFEMambaLocal":
    model = SEFEMambaLocal(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)
elif MODEL == "SEFE_PeriodicConv_MambaLocal":
    model = SEFE_PeriodicConv_MambaLocal(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)
elif MODEL == "SEFEMambaWithPeriod":
    # (Mamba, LearnablePeriod Estimator, SEFE, PeriodicConv, Mamba, LocalEnhancer)
    model = SEFEMambaWithPeriod(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)
elif MODEL == "SEFE_PeriodAwareMamba_MambaLocal": # 训练不稳定，试试initial方式的改变
    model = SEFE_PeriodAwareMamba_MambaLocal(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)

# 对比实验
elif MODEL == "MambaSEFEMamba":
    model = MambaSEFEMamba(input_dim=2, seq_len=seq_len, num_classes=num_classes).to(device)
else:
    print("ERROR ---> Please choose a model")

if DATASET == "Data-2016a_IQ":
    summary(model, (1, 2, 128))
elif DATASET == "Data-2018a_IQ":
    summary(model, (1, 2, 1024))


if DATASET == "Data-2016a_IQ":
    train_loader, val_loader, _ = datalodaer_2016a_IQ()
elif DATASET == "Data-2018a_IQ":
    train_loader, val_loader, _ = get_dataloader_2018a_IQ(batch_size=256)
else:
    train_loader, val_loader, _ = datalodaer_2016a_IQ()
NAME = 'Model[{}]_Data-[{}]'.format(MODEL, DATASET)

log_path = os.path.join(LOG_FOLDER, 'training_%s.log' % NAME)
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
# 清空已有 handlers
logger.handlers = []

# 添加新的 FileHandler
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# resume training or start new
start_epoch = 0
resumef = os.path.join(LOG_FOLDER, 'net_%s.pth' % NAME)
if os.path.isfile(resumef):
    checkpoint = torch.load(resumef)
    print("> Resuming previous training")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
    print("=> loaded parameters :")
    print("==> checkpoint['optimizer']['param_groups']")
    print("\t{}".format(checkpoint['optimizer']['param_groups']))
    print("==> checkpoint['training_params']")

step = 0  # 用于记录 batch 数
batch_losses = []  # 用于保存每 100 个 batch 的 loss

for epoch in range(start_epoch, num_epochs):
    if epoch < milestone:
        current_lr = lr
    else:
        current_lr = lr / 10.
    # set learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # === 每100个Batch记录一次Loss ===
        if step % 100 == 0:
            # print(f"Epoch [{epoch}] Step [{step}] Loss: {loss.item():.4f}")
            log_str = f"Epoch [{epoch}] Step [{step}] Loss: {loss.item():.4f}"
            print(log_str)
            logger.info(log_str)
            batch_losses.append((epoch, step, loss.item()))
        step += 1

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            # break # debug
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    log_str = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    print(log_str)
    logger.info(log_str)

    # save model and checkpoint
    save_dict = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(save_dict, os.path.join(LOG_FOLDER, 'net_%s.pth' % NAME))
file_handler.flush()
file_handler.close()
print("%s Training complete." % NAME)

if start_epoch == 0:
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    # 将 loss 按 epoch 分组
    epoch_loss_dict = defaultdict(list)
    for epoch, _, loss in batch_losses:
        epoch_loss_dict[epoch].append(loss)

    # 准备绘图数据
    epochs = sorted(epoch_loss_dict.keys())
    mean_losses = [np.mean(epoch_loss_dict[ep]) for ep in epochs]
    std_losses = [np.std(epoch_loss_dict[ep]) for ep in epochs]

    # 可视化
    plt.plot(epochs, mean_losses, label='Mean Training Loss', color='blue')
    plt.fill_between(epochs,
                     np.array(mean_losses) - np.array(std_losses),
                     np.array(mean_losses) + np.array(std_losses),
                     alpha=0.3, color='blue', label='±1 Std Dev')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch with Std Dev Range")

    # 绘制准确率图像
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)  # 设置y轴范围为 [0, 1.0]
    plt.yticks(np.arange(0, 1.01, 0.1))  # 设置y轴刻度间隔为0.1
    plt.legend()
    plt.title('Accuracy vs. Epochs')

    plt.tight_layout(pad=3.0)  # 增加子图之间的间距
    plt.savefig(os.path.join(LOG_FOLDER, "{}_Accuracy_vs_Epochs.png".format(NAME)))
    plt.close()
