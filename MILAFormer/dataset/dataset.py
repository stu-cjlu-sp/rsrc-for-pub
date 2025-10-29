import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm


def random_rotate(iq_signal, max_angle=np.pi):
    """
    对 IQ 信号进行随机旋转
    """
    # 随机生成旋转角度
    angle = np.random.uniform(-max_angle, max_angle)  # 随机选择角度，范围是 -π 到 π
    
    # 旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    
    # 对信号应用旋转矩阵
    rotated_signal = np.matmul(rotation_matrix, iq_signal)
    
    return rotated_signal

def zscore(X):
    #标准差归一化
    """
      X (ndarray): Shape (m,n) input data, m examples, n features
      X_norm (ndarray): Shape (m,n)  input normalized by column
      mu (ndarray):     Shape (n,)   mean of each feature
      sigma (ndarray):  Shape (n,)   standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return X_norm

# Load the modulation classes. You can also copy and paste the content of classes-fixed.txt.
class RML2018_random(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = path
        self.data = h5py.File(self.path, 'r')
        self.samples = self.data['samples']
        self.SNR = np.array(self.data['SNR'])
        self.label = self.data['label']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]


class RMLgeneral(Dataset):
    def __init__(self, samples, labels, SNR,max_angle=np.pi):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
        self.max_angle = max_angle 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq_signal = self.samples[idx]  # 形状为 (2, time_steps) 的信号
        rotated_iq_signal = random_rotate(iq_signal, self.max_angle)
        return torch.Tensor(rotated_iq_signal), self.SNR[idx], self.label[idx]
        #return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]
    
class RMLval(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]
    
class RMLtest(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
         return len(self.samples)

    def __getitem__(self, idx):

        return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]

