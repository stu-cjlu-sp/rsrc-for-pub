import torch.utils.data as data
import torchvision.transforms as tfs
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from PIL import Image
import glob
import scipy.io as scio

# dataset for training num
class Dataset_train_num(data.Dataset):##
    def __init__(self, path):
        self.path = path
        self.samples,self.classlabel = self.getpathitems(self.path)
    def getpathitems(self,path):
        all_imgs = []
        all_label = []
        samples = sorted(glob.glob(os.path.join(path + '/*')))
        for j in range(0,len(samples)):
            samples0=sorted(glob.glob(os.path.join(samples[j] + '/*')))

            for i in range(0,len(samples0)):
                samples1=sorted(glob.glob(os.path.join(samples0[i] + '/*')))
                for ii in range(0,len(samples1)):
                    img=sorted(glob.glob(os.path.join(samples1[ii] + '/*.*')))
                    all_imgs.append(img)
                    classlabel = np.ones(len(img),dtype=int) * j 
                    all_label.append(classlabel)

        all_imgs=np.array(all_imgs)
        all_imgs=np.reshape(all_imgs,-1)
        all_label=np.array(all_label)
        all_label=np.reshape(all_label,-1)

        return all_imgs,all_label
    def __getitem__(self, index):
        sample_source = Image.open(self.samples[index])
        data = tfs.ToTensor()(sample_source)
        classlabel = torch.tensor(self.classlabel[index])
        return data, classlabel

    def __len__(self):
        return len(self.samples) 

# dataset for training  two signal
class Dataset_train_two(data.Dataset):##
    '''
    Load images given the path
    '''
    def __init__(self, path,path_label):
        self.path = path
        self.path_label = path_label
        self.samples,self.classlabel = self.getpathitems(self.path,self.path_label)
    def getpathitems(self,path,path_label):
        target_img = scio.loadmat(path_label)
        classlabel = target_img['target']

        samples = sorted(glob.glob(os.path.join(path + '/*')))
        all_imgs = []
        all_label = []
        for i in range(0,len(samples)):
            samples1=sorted(glob.glob(os.path.join(samples[i] + '/*')))
            # if i == 1 or i == 2: # skip the -14dB and -12dB
            #     continue
            for ii in range(0,len(samples1)):
                img=sorted(glob.glob(os.path.join(samples1[ii] + '/*.*')))
                all_imgs.append(img)
            if i == 0:
                all_label = classlabel
            else:
                all_label=np.concatenate((all_label,classlabel),axis=0)
        all_imgs=np.array(all_imgs)
        all_imgs=np.reshape(all_imgs,-1)
        return all_imgs,all_label
    def __getitem__(self, index):
        sample_source = Image.open(self.samples[index])
        data = tfs.ToTensor()(sample_source)
        classlabel = torch.tensor(self.classlabel[index])
        return data, classlabel
    def __len__(self):
        return len(self.samples) 

# dataset for testing all
class Dataset_test_all(data.Dataset):##
    def __init__(self, path0, path1, path2,path_label):
        self.path0 = path0
        self.path1 = path1
        self.path2 = path2
        self.path_label = path_label
        self.samples, self.classlabel = self.getpathitems(self.path0, self.path1, self.path2, self.path_label)
    def getpathitems(self, path0, path1, path2,path_label):
        all_imgs = []
        all_label = []
        target_img = scio.loadmat(path_label)
        all_label = target_img['target']

        samples0 = sorted(glob.glob(os.path.join(path0 + '/*')))
        for j in range(0,len(samples0)):
            img=sorted(glob.glob(os.path.join(samples0[j] + '/*.*')))
            all_imgs.append(img)

        samples1 = sorted(glob.glob(os.path.join(path1 + '/*')))
        for j in range(0,len(samples1)):
            img=sorted(glob.glob(os.path.join(samples1[j] + '/*.*')))
            all_imgs.append(img)

        samples2 = sorted(glob.glob(os.path.join(path2 + '/*')))
        for j in range(0,len(samples2)):
            img=sorted(glob.glob(os.path.join(samples2[j] + '/*.*')))
            all_imgs.append(img)

        all_imgs=np.array(all_imgs)
        all_imgs=np.reshape(all_imgs,-1)

        return all_imgs,all_label
    def __getitem__(self, index):
        sample_source = Image.open(self.samples[index])
        data = tfs.ToTensor()(sample_source)
        classlabel = torch.tensor(self.classlabel[index])
        return data, classlabel

    def __len__(self):
        return len(self.samples) 

# dataset for testing num
class Dataset_test_num(data.Dataset):##
    '''
    Load images given the path
    '''
    def __init__(self, path0, path1, path2):
        self.path0 = path0
        self.path1 = path1
        self.path2 = path2
        self.samples, self.classlabel = self.getpathitems(self.path0, self.path1, self.path2)
    def getpathitems(self, path0, path1, path2):
        all_imgs = []
        all_label = []

        samples0 = sorted(glob.glob(os.path.join(path0 + '/*')))
        for j in range(0,len(samples0)):
            img=sorted(glob.glob(os.path.join(samples0[j] + '/*.*')))
            classlabel = np.zeros(len(img),dtype=int)
            all_label.append(classlabel)
            all_imgs.append(img)

        samples1 = sorted(glob.glob(os.path.join(path1 + '/*')))
        for j in range(0,len(samples1)):
            img=sorted(glob.glob(os.path.join(samples1[j] + '/*.*')))
            classlabel = np.ones(len(img),dtype=int) * 1
            all_label.append(classlabel)
            all_imgs.append(img)

        samples2 = sorted(glob.glob(os.path.join(path2 + '/*')))
        for j in range(0,len(samples2)):
            img=sorted(glob.glob(os.path.join(samples2[j] + '/*.*')))
            classlabel = np.ones(len(img),dtype=int) * 2
            all_label.append(classlabel)
            all_imgs.append(img)

        all_imgs=np.array(all_imgs)
        all_imgs=np.reshape(all_imgs,-1)
        all_label=np.array(all_label)
        all_label=np.reshape(all_label,-1)

        return all_imgs,all_label
    def __getitem__(self, index):
        sample_source = Image.open(self.samples[index])
        data = tfs.ToTensor()(sample_source)
        classlabel = torch.tensor(self.classlabel[index])
        return data, classlabel

    def __len__(self):
        return len(self.samples) 


    '''
    Load images given the path
    '''
    def __init__(self, path,path_label):
        self.path = path
        self.path_label = path_label
        self.samples,self.classlabel = self.getpathitems(self.path,self.path_label)
    def getpathitems(self,path,path_label):
        target_img = scio.loadmat(path_label)
        all_label = target_img['target']

        samples = sorted(glob.glob(os.path.join(path + '/*')))
        all_imgs = []
        for ii in range(0,len(samples)):
            img=sorted(glob.glob(os.path.join(samples[ii] + '/*.*')))
            all_imgs.append(img)

        all_imgs=np.array(all_imgs)
        all_imgs=np.reshape(all_imgs,-1)
        return all_imgs,all_label
    def __getitem__(self, index):
        sample_source = Image.open(self.samples[index])
        data = tfs.ToTensor()(sample_source)
        classlabel = torch.tensor(self.classlabel[index])
        return data, classlabel
    def __len__(self):
        return len(self.samples) 