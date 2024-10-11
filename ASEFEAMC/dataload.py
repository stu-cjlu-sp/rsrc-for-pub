import numpy as np
import pickle
from numpy import linalg as la
import numpy as np
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.layers import *
from keras import optimizers
from sklearn.preprocessing import StandardScaler

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
import h5py
import numpy as np
from keras.layers import Input,Conv1D,Dense,BatchNormalization,Activation,AveragePooling1D,GlobalAveragePooling1D,Lambda,MultiHeadAttention
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
import keras
import os

from keras import backend as K
from keras.layers import Permute, Dropout, Flatten, UpSampling1D
from keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN,Reshape
from keras.models import Model
import random

from resize import *

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
config= tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

## 导入数据
def data_load(snr):
    f = open('/home/sp604zq/zq/dual-channel/RML2016.10a_dict.pkl', 'rb') #导入路径
    data = pickle.load(f, encoding='latin1') 
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:                         
        X.append(data[(mod, snr)])           
    Y=np.zeros(shape=(11000,1))              
    for i in range(11):
        Y[i*1000:(i+1)*1000+1,]=i           
    X = np.vstack(X)                         
    return X,Y

## IQ
def generate_IQ(X):  
    X_IQ = np.zeros(shape=[220000, 2, 128])
    n = int(X.shape[0])
    
    for i in range(n):
        X_IQ[i, 0:2, :] = X[i]
    
    return X_IQ

## 归一化
def norm(X_IQ):
    n1=int(X_IQ.shape[0])
    n2=int(X_IQ.shape[1])
    X_norm=np.zeros(shape=[220000,n2,128])
    for i in range(n1):
        for j in range(n2):
            norm_num=1/la.norm(X_IQ[i,j,:],ord=np.inf)
            X_norm[i,j,:]=X_IQ[i,j,:]*norm_num
    resize_signal =  Resizesignal(prob=0.5)
    X_norm = resize_signal(X_norm)
    return X_norm

## 制作训练集与验证集
def train_test_split(data_snr,label_snr, test_ratio):
    random_state = 42                                       # 设置一个种子数
    np.random.seed(random_state)                            # 生成随机数
    shuffled_indices = np.random.permutation(len(data_snr))     # 生成随机序列
    test_set_size = int(len(data_snr) * test_ratio)             # 生成测试集大小
    test_indices = shuffled_indices[:test_set_size]             # 生成测试集
    train_indices = shuffled_indices[test_set_size:]            # 生成训练集
    return data_snr[train_indices],label_snr[train_indices], \
           data_snr[test_indices],label_snr[test_indices]