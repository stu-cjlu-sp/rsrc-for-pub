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
from model import *
from dataload import *
import random

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
config= tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def main():
    X1=[]
    Y1=[]
    for i in range(-20,20,2):
        X, Y = data_load(i)
        X1.append(X)
        Y1.append(Y)
    X1=np.array(X1)
    Y1=np.array(Y1)
    print(Y1.shape)
    Y=np.reshape(Y1,[220000,1])
    del Y1
    X1=np.reshape(X1,[220000,2,128])

    X=X1
    del X1

    Y=to_categorical(Y)
    print(Y.shape)

    X_IQ=generate_IQ(X)
    del X

    X_norm=norm(X_IQ) 

    del X_IQ
    X_norm=X_norm[:,0:2,:]  

    X_train,Y_train,X_test,Y_test=train_test_split(X_norm,Y,test_ratio=0.2)
    del X_norm
    # print(X_train.shape)   
    # print(Y_train.shape)   
    model = trymodel(MAX_SEQUENCE_LENGTH=128, NB_CLASS=11)

    filepath = '/home/sp604zq/zq/ASEFEAMC/model/ours+IQ+Resizesignal.h5'

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, min_lr=1e-7)

    early_stopping =EarlyStopping(monitor="val_loss", patience=18, verbose=1)
# ModelCheckpoint 回调
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
#adam = Adam(learning_rate=0.0007, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate=0.0007), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    model.fit([X_train], Y_train, epochs = 100, batch_size=64, verbose=1, shuffle=True, 
              validation_data=([X_test], Y_test), callbacks=[reduce_lr, early_stopping, checkpoint]) 


if __name__ == "__main__":
    main()

