import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from SignalQuantityDetection.cnn2 import *
from Dataload import *
import time
import torch.optim as optim
import torch
import torch.nn as nn
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from torchsummary import summary
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import BinaryRecall
import numpy as np
from torch.utils.data import DataLoader
from FocalLoss import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN(1,3)

# Focal Loss
alpha = [0.6,0.2,0.2]  # 每个类别的权重
loss_fn = FocalLoss(alpha=alpha, gamma=2, reduction='mean')

optimizer = optim.Adam(model.parameters(),lr=1E-3)
metric = MulticlassPrecision(num_classes=4).to(device)
recall = BinaryRecall().to(device)

def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt.cpu().detach().numpy(), predict.cpu().detach().numpy()), axis=1).astype("float32")
    return np.sum(ex_equal)

def train(model,optimizer,loss_fn,train_loader,epochs=50,device = "cpu"):    
    best_acc=0
    for epoch in range(epochs):   
        epoch_start_time = time.time()
        tot_correct=0.0 
        tot_loss=0.0
        tot_all = 0.0
        model.train()
        optimizer.zero_grad()
        for index,batch in enumerate(iter(train_loader)):
            training_loss=0.0
            preprocess_data,label = batch
            preprocess_data = preprocess_data.cuda()
            model = model.cuda()
            result = model(preprocess_data)

            loss = loss_fn(result.cpu(),label.long())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predict_y = torch.argmax(result.cpu(),1).data.numpy()
            tot_correct += sum(predict_y==label.cpu().data.numpy())
            tot_all += len(predict_y)
            tot_loss += loss.item()*len(predict_y)
        training_loss /= len(train_loader)
        
        avg_acc = tot_correct/tot_all
        loss_epoch = tot_loss/tot_all
        print('[Epoch %d]lpinet time:%.1f, avg_loss:%.4f,avg_acc:%.4f'%(epoch+1,time.time() - epoch_start_time,loss_epoch,avg_acc))

        if avg_acc > best_acc:   
            print("已修改模型")
            best_acc = avg_acc
            torch.save(model, "/home/sp604yky/yky/SignalQuantityDetection/rec_num_100.pth")


if __name__ == '__main__':
    root_data = '/home/sp604yky/yky/DCNN/data/train'
    CWD_train_loader=DataLoader(dataset=Dataset_train_num(path=root_data),batch_size=256,shuffle=True)
    train(model,optimizer,loss_fn,CWD_train_loader,epochs=100)

