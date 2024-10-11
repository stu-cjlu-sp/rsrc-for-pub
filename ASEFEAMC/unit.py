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


import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
config= tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# 挤压激励模块
def SEBlock1D(input_layer, reduction_ratio): #通过挤压操作学习每个通道的权重从而加强重要通道的特征 
    # 获取输入张量的形状
    input_shape = K.int_shape(input_layer)  # 使用 K.int_shape() 函数获取张量形状

    # 挤压操作
    squeeze = GlobalAveragePooling1D()(input_layer)
    squeeze = Dense(int(input_shape[-1] // reduction_ratio), activation='relu')(squeeze)
    squeeze = Dense(int(input_shape[-1]), activation='sigmoid')(squeeze)
    squeeze = Reshape((1, input_shape[-1]))(squeeze)

    # 通道加权
    scaled_input = multiply([input_layer, squeeze])

    return scaled_input


# 定义位置嵌入
def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''
    ''' 初始化正弦编码表 '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1  #这实际是使用了正弦函数和余弦函数来为序列中的不同位置分配不同的编码值
    
    return position_enc


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)