from keras.models import load_model
from sklearn.metrics import accuracy_score
import os
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix
from aff_resnet import ResNet_block, BasicBlock
from fusion import AFF, iAFF, DAF
from transformer_mode import SEBlock

import matplotlib.pyplot as plt
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_data_train(directory_name,  then_file):
    train_data1 = []
    train_label1 = []
    i = -1
    for last_file1 in os.listdir(directory_name + '/' + then_file):
        i = i + 1
        last_allfile = ['LFM', 'Costas', 'BPSK', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
        last_file = last_allfile[i]
        for filename in os.listdir(directory_name + '/' + then_file + '/' + last_file):
            mat_file = h5py.File(directory_name + '/' +  then_file  + '/' +last_file + '/' + filename)
            data_key = list(mat_file)
            train_data = mat_file[data_key[0]]
            if train_data.shape[1]>=1024:
                train_data = train_data[:, :1024] 
            else: 
                a = 1024-train_data.shape[1]
                train_data =np.concatenate((train_data, np.zeros((train_data.shape[0], a))), axis=1)                                                                 
            train_label1.append(i)
            train_data1.append(train_data)
    train_data1 = np.array(train_data1)
    train_label1 = np.array(train_label1)
    return train_data1, train_label1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

snrs = range(-14, 22, 2)
ac = {}
for SNR in snrs:
    db = str(SNR)+'db'
    train_image2, train_label2 = load_data_train(directory_name='Data/300_1/test2', then_file=db)
    idx = np.random.permutation(train_image2.shape[0])
    train_image2,train_label2 = train_image2[idx], train_label2[idx]
    custom_objects = {
    'ResNet_block': ResNet_block, 
    'BasicBlock': BasicBlock,  
    'AFF': AFF, 
    'iAFF': iAFF, 
    'DAF': DAF,
    'SEBlock': SEBlock
    }
    model = load_model('fine-tuning_weights/base_mode_resnet18_AFF_coordatt.h5', custom_objects=custom_objects)
    a = np.argmax(model.predict(train_image2), axis=1) 
    print(" classification results:\n", a)
    AC = accuracy_score(train_label2, a)    
    print("SNR:", SNR, "AC", AC)
    ac[SNR] = AC 
print(ac)
