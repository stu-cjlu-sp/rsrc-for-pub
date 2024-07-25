import torch
import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import scipy.io as scio
import h5py
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

traindata_root = './data/trainsigal.mat'
trainlabel_root = './data/trainlabel.mat'
trainnote_root = './data/MSSTtrainlabel.mat'

valdata_root = './data/valsigal.mat'
vallabel_root = './data/vallabel.mat'
valnote_root = './data/MSSTvallabel.mat'

class traindata(data_utils.Dataset):
    def __init__(self,datapath,labelpath,notepath) -> None:
        super(traindata).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        noisy_sig = scio.loadmat(datapath)
        target_img = scio.loadmat(labelpath)
        note = scio.loadmat(notepath)
        self.noisy_sig = noisy_sig['signal']
        self.target_img = target_img['target']
        self.note = torch.tensor(note['label'])

        target_img = torch.tensor(self.target_img)
        target_img = np.array(target_img).astype(np.float32)
        self.realdata = np.real(self.noisy_sig)
        self.imagdata = np.imag(self.noisy_sig)
        self.bz = self.noisy_sig.shape[0]
        N=self.noisy_sig.shape[1]
        self.noisy_sig = np.zeros([int(self.bz), 2, N]).astype(np.float32)
        self.noisy_sig[:, 0,:] = (self.realdata.astype(np.float32))
        self.noisy_sig[:, 1,:] = (self.imagdata.astype(np.float32))
    def __getitem__(self, index):
        noisy_sig = self.noisy_sig[index]
        target_img = self.transform(self.target_img[index])
        target_img4 = np.array(target_img)
        target_img3 = (target_img4 - np.min(target_img4)) / (np.max(target_img4) - np.min(target_img4))
        notes = self.note[index]
        return noisy_sig, target_img3, notes
    def __len__(self):
        return len(self.noisy_sig)

torch_data1 = traindata(traindata_root,trainlabel_root,trainnote_root)
traindatas = DataLoader(torch_data1, batch_size=32 ,shuffle=True, drop_last=False)

class valdata(data_utils.Dataset):
    def __init__(self,datapath,labelpath,notepath) -> None:
        super(traindata).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        noisy_sig = scio.loadmat(datapath)
        target_img = scio.loadmat(labelpath)
        note = scio.loadmat(notepath)        
        self.noisy_sig = noisy_sig['signal']
        self.target_img = target_img['target']
        self.note = torch.tensor(note['label'])

        target_img = torch.tensor(self.target_img)
        target_img = np.array(target_img).astype(np.float32)
        self.realdata = np.real(self.noisy_sig)
        self.imagdata = np.imag(self.noisy_sig)
        self.bz = self.noisy_sig.shape[0]
        N=self.noisy_sig.shape[1]
        self.noisy_sig = np.zeros([int(self.bz), 2, N]).astype(np.float32)
        self.noisy_sig[:, 0,:] = (self.realdata.astype(np.float32))
        self.noisy_sig[:, 1,:] = (self.imagdata.astype(np.float32))
    def __getitem__(self, index):
        noisy_sig = self.noisy_sig[index]
        target_img = self.transform(self.target_img[index])
        target_img4 = np.array(target_img)
        target_img3 = (target_img4 - np.min(target_img4)) / (np.max(target_img4) - np.min(target_img4))
        notes = self.note[index]
        return noisy_sig, target_img3, notes
    def __len__(self):
        return len(self.noisy_sig)
    
torch_data2 = valdata(valdata_root, vallabel_root, valnote_root)
valdatas = DataLoader(torch_data2, batch_size=32, shuffle=True, drop_last=False)

