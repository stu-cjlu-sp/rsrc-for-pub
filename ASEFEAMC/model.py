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
from unit import *
import random


import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
config= tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def trymodel(MAX_SEQUENCE_LENGTH,  NB_CLASS):
    ip = Input(shape=(2,MAX_SEQUENCE_LENGTH))     
    y1 = Permute((2, 1))(ip)                       

    y2 = Bidirectional(LSTM(32,return_sequences=True))(y1) #保证输出的序列大小不变
    y2 = Dropout(0.2)(y2) 
  
    y3 = Conv1D(32, 7, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y2)         
    y3 = BatchNormalization()(y3)                 # 进行批归一化操作
    y4 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y2)         
    y4 = BatchNormalization()(y4)    
    y5 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 2)(y2)         
    y5 = BatchNormalization()(y5) 

    y6 = concatenate([y2,y3,y4,y5])
    y7 = Conv1D(32, 1, padding='same',activation='relu',kernel_initializer='he_uniform')(y6)
    y7 = BatchNormalization()(y7)

    y7 = SEBlock1D(y7, reduction_ratio=1)

    y8 = Conv1D(32, 7, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y7)         
    y8 = BatchNormalization()(y8)                 # 进行批归一化操作
    y9 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y7)         
    y9 = BatchNormalization()(y9)    
    y10 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 2)(y7)         
    y10 = BatchNormalization()(y10) 

    y11 = concatenate([y7,y8,y9,y10])
    y12 = Conv1D(32, 1, padding='same',activation='relu',kernel_initializer='he_uniform')(y11)
    y12 = BatchNormalization()(y12)

    y12 = SEBlock1D(y12, reduction_ratio=1)

    y13 = Conv1D(32, 7, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y12)         
    y13 = BatchNormalization()(y13)                 # 进行批归一化操作
    y14 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y12)         
    y14 = BatchNormalization()(y14)    
    y15 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 2)(y12)         
    y15 = BatchNormalization()(y15) 

    y16 = concatenate([y12,y13,y14,y15])
    y17 = Conv1D(32, 1, padding='same',activation='relu',kernel_initializer='he_uniform')(y16)
    y17 = BatchNormalization()(y17)

    y17 = SEBlock1D(y17, reduction_ratio=1) 

    y18 = Conv1D(32, 7, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y17)         
    y18 = BatchNormalization()(y18)                 # 进行批归一化操作
    y19 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y17)         
    y19 = BatchNormalization()(y19)    
    y20 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 2)(y17)         
    y20 = BatchNormalization()(y20) 
  
    y21 = concatenate([y17,y18,y19,y20])
    y22 = Conv1D(32, 1, padding='same',activation='relu',kernel_initializer='he_uniform')(y21)
    y22 = BatchNormalization()(y22)

    y22 = SEBlock1D(y22, reduction_ratio=1) 

    y23 = Conv1D(32, 7, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y22)         
    y23 = BatchNormalization()(y23)                 # 进行批归一化操作
    y24 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y22)         
    y24 = BatchNormalization()(y24)    
    y25 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 2)(y22)         
    y25 = BatchNormalization()(y25) 

    y26 = concatenate([y22,y23,y24,y25])
    y27 = Conv1D(32, 1, padding='same',activation='relu',kernel_initializer='he_uniform')(y26)
    y27 = BatchNormalization()(y27)

    y27 = SEBlock1D(y27, reduction_ratio=1) 

    y28 = Conv1D(32, 7, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y27)         
    y28 = BatchNormalization()(y28)                 # 进行批归一化操作
    y29 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 3)(y27)         
    y29 = BatchNormalization()(y29)    
    y30 = Conv1D(32, 3, padding='same',activation='relu',kernel_initializer='he_uniform',dilation_rate = 2)(y27)         
    y30 = BatchNormalization()(y30) 
 
    y31 = concatenate([y27,y28,y29,y30])
    y32 = Conv1D(32, 1, padding='same',activation='relu',kernel_initializer='he_uniform')(y31)
    y32 = BatchNormalization()(y32)

    y32 = SEBlock1D(y32, reduction_ratio=1) 

    pos_emb = position_encoding_init(128,32)
    y_pos = y32 + pos_emb
    
    transformer_block = TransformerBlock(32, 2, 32)
    
    y48 = transformer_block(y_pos)

    #池化层
    y49 = GlobalAveragePooling1D()(y48) #如果要应用到t-sne算法 那么要取的是这一层

    y50= Dense(16,activation='relu')(y49)
    
    # Fully connected layer
    out = Dense(NB_CLASS, activation='softmax')(y50)
    model = Model(ip, out)
    model.summary() 

    return model