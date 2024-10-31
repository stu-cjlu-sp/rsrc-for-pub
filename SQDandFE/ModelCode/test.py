import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Dataload import *
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import BinaryRecall
import torch
import numpy as np

weights_path = '/home/sp604yky/yky/SignalQuantityDetection/model/ensemble_weight_622.pth'
if os.path.exists(weights_path):
    weights = torch.load(weights_path)
else:
    weights = torch.tensor([1/3, 1/3, 1/3])

model_num1 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/num_model_100.pth')
model_num2 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/num_model_010.pth')
model_num3 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/num_model_001.pth')
model_1 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/rec_model_s1.pth')
model_2 = torch.load ('/home/sp604yky/yky/SignalQuantityDetection/model/rec_model_s2.pth')

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

metric = MulticlassPrecision(num_classes=4).to(device)
recall = BinaryRecall().to(device)

def example_subset_accuracy(gt, predict):

    ex_equal = np.all(np.equal(gt.cpu().detach().numpy(), predict.cpu().detach().numpy()), axis=1).astype("float32")
    return np.sum(ex_equal)


def test(model_num1,model_num2,model_num3,model_1,model_2,test_loader,snr,device = "cpu"):
    model_num1.eval()
    model_num2.eval()
    model_num3.eval()
    model_1.eval()
    model_2.eval()
    tot_true=0.0
    tot_all=0.0

    for index,batch in enumerate(iter(test_loader)):
        preprocess_data,label = batch
        preprocess_data = preprocess_data.cuda()
        output1= model_num1(preprocess_data)
        output2= model_num2(preprocess_data)
        output3= model_num3(preprocess_data)  

        output = weights[0]*output1 + weights[1]*output2 + weights[2]*output3
        predict_y = torch.argmax(output.cpu(),1).data.numpy()
        x_prob_1, x_thre_1 = model_1(preprocess_data)
        pre_result_1 = x_prob_1 - x_thre_1
        result_1 = torch.zeros_like(pre_result_1)
        _, indices = torch.topk(pre_result_1, k=1, dim=1)
        result_1.scatter_(1, indices, 1)

        x_prob_2, x_thre_2 = model_2(preprocess_data)
        pre_result_2 = x_prob_2 - x_thre_2
        result_2 = torch.zeros_like(pre_result_2)
        _, indices = torch.topk(pre_result_2, k=2, dim=1)
        result_2.scatter_(1, indices, 1)
        result_all = torch.zeros((len(label),4))
        for i in range(len(label)):
            if predict_y[i] == 1:
                result_all[i] = result_1[i]
            elif predict_y[i] == 2:
                result_all[i] = result_2[i]
            else:
                result_all[i] = torch.zeros((1,4))
        
        true_pre = example_subset_accuracy(result_all, label)
        tot_all += len(label)
        tot_true += true_pre

    avg_acc = tot_true/tot_all
    return avg_acc

if __name__ == '__main__':
    acc = []
    root_target = '/home/sp604yky/yky/DCNN/data/target_200_all.mat'
    for snr in range(-14,12,2): 
        root_data0 = f'/home/sp604yky/yky/DCNN/data/test/test_3_0/{snr}db'
        root_data1 = f'/home/sp604yky/yky/DCNN/data/test/test_3_1/{snr}db'
        root_data2 = f'/home/sp604yky/yky/DCNN/data/test/test_3_2/{snr}db'
        CWD_test_loader=DataLoader(dataset = Dataset_test_all(root_data0,root_data1,root_data2,root_target),batch_size=8,shuffle=True)
        avg_acc = test(model_num1,model_num2,model_num3,model_1,model_2,CWD_test_loader,snr)
        print('SNR = %.0f dB: acc = %.4f' %(snr,avg_acc))
