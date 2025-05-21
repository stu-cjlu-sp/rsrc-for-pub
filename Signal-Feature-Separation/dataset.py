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
trainclasslabel_root1 = './traindata/trainclasslabel1'
trainclasslabel_root2 = './traindata/trainclasslabel2'

valdata_root = './valdata/valsignal.mat'
valclasslabel_root1 = './valdata/vallabel1.mat'
valclasslabel_root2 = './valdata/vallabel2.mat'

class traindata(data_utils.Dataset):
    def __init__(self,datapath,classlabelpath1,classlabelpath2) -> None:
        super(traindata).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        samples_data = sorted(glob.glob(os.path.join(datapath + '/*.*')))
        samples_classlabel1 = sorted(glob.glob(os.path.join(classlabelpath1+ '/*.*')))
        samples_classlabel2 = sorted(glob.glob(os.path.join(classlabelpath2+ '/*.*')))
        for i in range(0,len(samples_data)):
            data=scio.loadmat(samples_data[i])
            classlabel1=scio.loadmat(samples_classlabel1[i])
            classlabel2=scio.loadmat(samples_classlabel2[i])
            data = data['aa']
            classlabel1 = classlabel1['cc1']
            classlabel2 = classlabel2['cc2']
            if i == 0:
                all_data = data
                all_classlabel1 = classlabel1
                all_classlabel2 = classlabel2
            else:
                all_data = np.concatenate((all_data, data), axis=0)
                all_classlabel1 = np.concatenate((all_classlabel1, classlabel1), axis=0)
                all_classlabel2 = np.concatenate((all_classlabel2, classlabel2), axis=0)

        self.noisy_sig = all_data
        self.label1 = all_classlabel1
        self.label2 = all_classlabel2


        self.realdata = np.real(self.noisy_sig)
        self.imagdata = np.imag(self.noisy_sig)
        self.bz = self.noisy_sig.shape[0]
        N=self.noisy_sig.shape[1]
        self.noisy_sig = np.zeros([int(self.bz), 2, N]).astype(np.float32)
        self.noisy_sig[:, 0,:] = (self.realdata.astype(np.float32))
        self.noisy_sig[:, 1,:] = (self.imagdata.astype(np.float32))
    def __getitem__(self, index):
        noisy_sig = self.noisy_sig[index]
        classlabel1 = self.label1[index]
        classlabel2 = self.label2[index]
        return noisy_sig, classlabel1, classlabel2
    def __len__(self):
        return len(self.noisy_sig)

torch_data1 = traindata(traindata_root, trainclasslabel_root1, trainclasslabel_root2)
traindatas = DataLoader(torch_data1, batch_size=16 ,shuffle=True, drop_last=False)


class valdata(data_utils.Dataset):
    def __init__(self,datapath,classlabelpath1,classlabelpath2) -> None:
        super(valdata).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        noisy_sig = scio.loadmat(datapath)
        classlabel1 = scio.loadmat(classlabelpath1)
        classlabel2 = scio.loadmat(classlabelpath2)

        self.noisy_sig = noisy_sig['aa']
        self.classlabel1 = torch.tensor(classlabel1['cc1'])
        self.classlabel2 = torch.tensor(classlabel2['cc2'])

        self.realdata = np.real(self.noisy_sig)
        self.imagdata = np.imag(self.noisy_sig)
        self.bz = self.noisy_sig.shape[0]
        N=self.noisy_sig.shape[1]
        self.noisy_sig = np.zeros([int(self.bz), 2, N]).astype(np.float32)
        self.noisy_sig[:, 0,:] = (self.realdata.astype(np.float32))
        self.noisy_sig[:, 1,:] = (self.imagdata.astype(np.float32))
    def __getitem__(self, index):
        noisy_sig = self.noisy_sig[index]
        classlabel1 = self.classlabel1[index]
        classlabel2 = self.classlabel2[index]
        return noisy_sig, classlabel1, classlabel2
    def __len__(self):
        return len(self.noisy_sig)
      
torch_data2 = valdata(valdata_root, valclasslabel_root1, valclasslabel_root2)
valdatas = DataLoader(torch_data2, batch_size=16, shuffle=True, drop_last=False)

