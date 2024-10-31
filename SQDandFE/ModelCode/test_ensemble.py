import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# from LPInet import *
from Dataload import Dataset_test_num
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import BinaryRecall

import torch.optim as optim
import torch
import numpy as np
import time

weights_path = '/home/sp604yky/yky/SignalQuantityDetection/model/ensemble_weight_622.pth'
if os.path.exists(weights_path):
    weights = torch.load(weights_path)
else:
    weights = torch.tensor([1/3, 1/3, 1/3])
model_1 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/num_model_100.pth')
model_2 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/num_model_010.pth')
model_3 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/num_model_001.pth')
 
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

metric = MulticlassPrecision(num_classes=4).to(device)
recall = BinaryRecall().to(device)

def example_subset_accuracy(gt, predict):

    ex_equal = np.all(np.equal(gt.cpu().detach().numpy(), predict.cpu().detach().numpy()), axis=1).astype("float32")
    return np.sum(ex_equal)


def test(model_1,model_2,model_3,test_loader,weights,device = "cpu"):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    tot_correct=0.0
    tot_all=0.0
    for index,batch in enumerate(iter(test_loader)):
        preprocess_data,label = batch
        preprocess_data = preprocess_data.cuda()
        label = label.cuda()
        output1= model_1(preprocess_data)
        output2= model_2(preprocess_data)
        output3= model_3(preprocess_data)  

        output = weights[0]*output1 + weights[1]*output2 + weights[2]*output3

        predict_y = torch.argmax(output.cpu(),1).data.numpy()
        tot_correct += sum(predict_y==label.cpu().data.numpy())
        tot_all += len(predict_y)
    avg_acc = tot_correct/tot_all
    return avg_acc

if __name__ == '__main__':
    acc = []
    for snr in range(2,12,2): 
        root_data0 = f'/home/sp604yky/yky/DCNN/data/test/test_3_0/{snr}db'
        root_data1 = f'/home/sp604yky/yky/DCNN/data/test/test_3_1/{snr}db'
        root_data2 = f'/home/sp604yky/yky/DCNN/data/test/test_3_2/{snr}db'
        CWD_test_loader=DataLoader(dataset = Dataset_test_num(root_data0,root_data1,root_data2),batch_size=128,shuffle=True)
        avg_acc = test(model_1,model_2,model_3,CWD_test_loader,weights)
        print('SNR = %.0f dB: acc = %.4f' %(snr, avg_acc))
