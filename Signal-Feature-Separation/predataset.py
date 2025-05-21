import torch
import os
import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import scipy.io as scio
import h5py
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import glob

traindata_root = './traindata/trainsig'
trainMSSTlabel_root1 = './traindata/trainMSSTlabel1'
trainMSSTlabel_root2 = './traindata/trainMSSTlabel2'

valdata_root = './valdata/valsignal.mat'
valMSSTlabel_root1 = './valdata/valMSSTlabel1.mat'
valMSSTlabel_root2 = './valdata/valMSSTlabel2.mat'



class traindata(data_utils.Dataset):
    def __init__(self,datapath,MSSTlabelpath1,MSSTlabelpath2) -> None:
        super(traindata).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        samples_data = sorted(glob.glob(os.path.join(datapath + '/*.*')))
        samples_msst1 = sorted(glob.glob(os.path.join(MSSTlabelpath1+ '/*.*')))
        samples_msst2 = sorted(glob.glob(os.path.join(MSSTlabelpath2+ '/*.*')))
        for i in range(0,len(samples_data)):
            data=scio.loadmat(samples_data[i])
            msst1=scio.loadmat(samples_msst1[i])
            msst2=scio.loadmat(samples_msst2[i])
            data = data['aa']
            msst1 = msst1['bb1']
            msst2 = msst2['bb2']
            if i == 0:
                all_data = data
                all_msst1 = msst1
                all_msst2 = msst2
            else:
                all_data = np.concatenate((all_data, data), axis=0)
                all_msst1 = np.concatenate((all_msst1, msst1), axis=0)
                all_msst2 = np.concatenate((all_msst2, msst2), axis=0)

        self.noisy_sig = all_data
        self.target_img1 = all_msst1
        self.target_img2 = all_msst2

        self.realdata = np.real(self.noisy_sig)
        self.imagdata = np.imag(self.noisy_sig)
        self.bz = self.noisy_sig.shape[0]
        N=self.noisy_sig.shape[1]
        self.noisy_sig = np.zeros([int(self.bz), 2, N]).astype(np.float32)
        self.noisy_sig[:, 0,:] = (self.realdata.astype(np.float32))
        self.noisy_sig[:, 1,:] = (self.imagdata.astype(np.float32))
    def __getitem__(self, index):
        noisy_sig = self.noisy_sig[index]
        target_imga = self.transform(self.target_img1[index])
        target_imgb = np.array(target_imga)
        target_img1 = (target_imgb - np.min(target_imgb)) / (np.max(target_imgb) - np.min(target_imgb))
        
        target_imgc = self.transform(self.target_img2[index])
        target_imgd = np.array(target_imgc)
        target_img2 = (target_imgd - np.min(target_imgd)) / (np.max(target_imgd) - np.min(target_imgd))

        return noisy_sig, target_img1, target_img2
    def __len__(self):
        return len(self.noisy_sig)

torch_data1 = traindata(traindata_root,trainMSSTlabel_root1,trainMSSTlabel_root2)
traindatas = DataLoader(torch_data1, batch_size=32 ,shuffle=True, drop_last=False)


class valdata(data_utils.Dataset):
    def __init__(self,datapath,MSSTlabelpath1,MSSTlabelpath2) -> None:
        super(valdata).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        noisy_sig = scio.loadmat(datapath)
        MSSTlabel1 = scio.loadmat(MSSTlabelpath1)
        MSSTlabel2 = scio.loadmat(MSSTlabelpath2)
        self.noisy_sig = noisy_sig['aa']
        self.MSSTlabel1 = MSSTlabel1['bb1']
        self.MSSTlabel2 = MSSTlabel2['bb2']

        MSSTlabel1 = torch.tensor(self.MSSTlabel1)
        MSSTlabel1 = np.array(MSSTlabel1).astype(np.float32)

        MSSTlabel2 = torch.tensor(self.MSSTlabel2)
        MSSTlabel2 = np.array(MSSTlabel2).astype(np.float32)

        self.realdata = np.real(self.noisy_sig)
        self.imagdata = np.imag(self.noisy_sig)
        self.bz = self.noisy_sig.shape[0]
        N=self.noisy_sig.shape[1]
        self.noisy_sig = np.zeros([int(self.bz), 2, N]).astype(np.float32)
        self.noisy_sig[:, 0,:] = (self.realdata.astype(np.float32))
        self.noisy_sig[:, 1,:] = (self.imagdata.astype(np.float32))
    def __getitem__(self, index):
        noisy_sig = self.noisy_sig[index]
        target_imga = self.transform(self.MSSTlabel1[index])
        target_imgb = np.array(target_imga)
        target_img1 = (target_imgb - np.min(target_imgb)) / (np.max(target_imgb) - np.min(target_imgb))
        
        target_imgc = self.transform(self.MSSTlabel2[index])
        target_imgd = np.array(target_imgc)
        target_img2 = (target_imgd - np.min(target_imgd)) / (np.max(target_imgd) - np.min(target_imgd))

        return noisy_sig, target_img1, target_img2
    def __len__(self):
        return len(self.noisy_sig)
      
torch_data2 = valdata(valdata_root, valMSSTlabel_root1, valMSSTlabel_root2)
valdatas = DataLoader(torch_data2, batch_size=8, shuffle=True, drop_last=False)

