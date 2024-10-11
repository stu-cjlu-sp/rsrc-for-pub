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

# 缩放0.8-1.2倍左右
class Resizesignal(object):
    """ 对输入的信号进行放缩操作 """
    def __init__(self,prob: float = 0.5):
        self.prob = prob
        
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < self.prob:
            sample_return = sample.copy()
            s = random.randint(8,14)
            I = sample[0, :] * s * 0.1
            Q = sample[1, :] * s * 0.1
            sample_return[0, :] = I.copy()
            sample_return[1, :] = Q.copy()
            return sample_return
        return sample