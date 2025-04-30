import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchinfo import summary
from model import VSSM
from model_v0 import VSSM_Flatten
from utils import get_dataloader_2016a_TF, get_dataframe

"""
######################################
Training code of time frequency data
######################################
"""

# Settings
SNR_list = [n for n in range(-20, 20, 2)]  # list of SNRs: [-20,-18....,16,18]  2016a dataset
# SNR_list = [0] # for debug
MODEL = "FlattenMamba" # "MedMamba"
DATASET = "Data-2016a_TF"
LOG_FOLDER = "logs/{}_{}".format(DATASET, MODEL)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

num_epochs = 100
milestone = 30 # for learning rate
lr = 0.001

for snr in SNR_list:
    train_loader, val_loader, test_loader = get_dataloader_2016a_TF(snr, batch_size=32)
    NAME = 'Model[{}]_SNR[{}]_Data-[{}]'.format(MODEL, str(snr), DATASET)
    # reset logging
    logging.getLogger().handlers = []
    # Configure logging
    logging.basicConfig(filename=os.path.join(LOG_FOLDER, 'training_%s.log' % NAME), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 设置模型运行的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if MODEL == "MedMamba":
        model = VSSM(patch_size=4, in_chans=1, depths=[2, 2, 12, 2], dims=[64, 128, 256, 512], num_classes=11).to(device)
    elif MODEL == "FlattenMamba":
        model = VSSM_Flatten(in_chans=1, depths=[2, 2, 4, 2], dims=[128, 128, 128, 128], num_classes=11, d_state=16, patch_size=4).to("cuda")
    else:
        print("Please choose a model")
    summary(model, (1, 1, 224, 224))
    print("############## Input SNR: {}  ##############".format(snr))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        val_loss = running_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        log_str = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        print(log_str)
        logging.info(log_str)

        # save model and checkpoint
        save_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_dict, os.path.join(LOG_FOLDER, 'net_%s.pth' % NAME))

    print("Training complete.")

    if start_epoch == 0:
        epochs = range(1, num_epochs + 1)

        plt.figure(figsize=(12, 5))

        # 绘制损失图像
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epochs')

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

        # plt.show()
        plt.savefig(os.path.join(LOG_FOLDER, "{}_Accuracy_vs_Epochs.png".format(NAME)))


        all_labels = []
        all_predictions = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)  # 确保输入数据在正确的设备上
                labels = labels.to(device)  # 确保标签也在正确的设备上
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())  # 将数据移回 CPU 并转换为 numpy
                all_predictions.extend(predicted.cpu().numpy())  # 将数据移回 CPU 并转换为 numpy

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        data_frame = get_dataframe(snr=snr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_frame['Mod_Type'].astype('category').cat.categories)

        # 设置显示尺寸并绘制混淆矩阵
        plt.figure(figsize=(15, 20))  # 可以调整尺寸以适应你的需求
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        # plt.show()
        plt.savefig(os.path.join(LOG_FOLDER, "{}_Confusion_Matrix.png".format(NAME)))
