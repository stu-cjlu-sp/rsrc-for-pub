import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from Dataload import *
import time
import torch.optim as optim
import torch
import torch.nn as nn
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import BinaryRecall
import numpy as np
from torch.utils.data import DataLoader
from aff_resnet_copy import *

softmax = nn.Softmax(dim=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN(1)

optimizer = optim.Adam(model.parameters(),lr=1E-3)
loss_fn = nn.BCEWithLogitsLoss()
metric = MulticlassPrecision(num_classes=4).to(device)
recall = BinaryRecall().to(device)

def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt.cpu().detach().numpy(), predict.cpu().detach().numpy()), axis=1).astype("float32")
    return np.sum(ex_equal)

def train(model,optimizer,loss_fn,train_loader,signal_num,epochs=50,device = "cpu"):    
    best_acc=9999
    for epoch in range(epochs):   
        epoch_start_time = time.time()
        tot_correct=0.0
        tot_loss=0.0
        tot_all = 0.0
        model.train()
        optimizer.zero_grad()
        for index,batch in enumerate(iter(train_loader)):
            preprocess_data,label = batch
            preprocess_data = preprocess_data.cuda()
            inverted_label = torch.where(label == 0, torch.tensor(1), torch.tensor(0))
            inverted_label = torch.tensor(inverted_label)
            label = label.cuda()
            inverted_label =inverted_label.cuda()
            model = model.cuda()
            x_prob, x_thre = model(preprocess_data)

            loss1 = loss_fn(x_prob,label.float())
            loss2 = loss_fn(x_thre,inverted_label.float())
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_loss += loss.item()*len(label)
            
            pre_result = x_prob - x_thre
            result = torch.zeros_like(pre_result)
            _, indices = torch.topk(pre_result, k=signal_num, dim=1)
            result.scatter_(1, indices, 1)

            ex_equal = example_subset_accuracy(result,label)
            tot_correct += np.sum(ex_equal)
            tot_all += len(label)
        acc_epoch = tot_correct / tot_all
        loss_epoch = tot_loss/tot_all

        print('[Epoch %d]lpinet time:%.1f, avg_loss:%.4f, acc:%.4f'%(epoch+1,time.time() - epoch_start_time,loss_epoch,acc_epoch))

        if loss_epoch < best_acc:  
            print("已修改模型")
            best_acc = loss_epoch
            save_path = f"/home/sp604yky/yky/SignalQuantityDetection/rec_model_s{signal_num}.pth"
            torch.save(model, save_path)


if __name__ == '__main__':
    signal_num = 1
    root_data = f'/home/sp604yky/yky/DCNN/data/train/train_3_{signal_num}'
    root_target = f'/home/sp604yky/yky/DCNN/data/target_400_{signal_num}.mat'
    CWD_train_loader=DataLoader(dataset=Dataset_train_two(path=root_data, path_label = root_target),batch_size=256,shuffle=True)
    train(model,optimizer,loss_fn,CWD_train_loader,signal_num,epochs=100)

