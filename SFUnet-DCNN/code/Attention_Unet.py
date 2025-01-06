import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from load_dataset import *
from cosine_annealing import CosineAnnealingScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
from keras_flops import get_flops



def attention(deep_input, shallow_input):
    
    au1 = Conv2D(filters=1,kernel_size=1,strides=1,padding="same")(deep_input)
    au1_up = UpSampling2D(size=2)(au1)
    au2 = Conv2D(filters=1,kernel_size=1,strides=1,padding="same")(shallow_input)
    au3 = Add()([au1_up,au2])
    au4 = Activation('relu')(au3)
    au5 = Conv2D(filters=1,kernel_size=1,strides=1,padding="same")(au4)
    au6 = Activation('sigmoid')(au5)
    au7 = Multiply()([au6, shallow_input])

    return au7


def unet(input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)


    au1 = attention(conv4, conv3)
    up1 = UpSampling2D(size = (2,2))(conv4)
    up1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up1)
    merge1 = concatenate([au1, up1], axis = 3)

    au2 = attention(up1, conv2)
    up2 = UpSampling2D(size = (2,2))(merge1)
    up2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up2)
    merge2 = concatenate([au2, up2], axis = 3)

    au3 = attention(up2, conv1)
    up3 = UpSampling2D(size = (2,2))(merge2)
    up3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up3)
    merge3 = concatenate([au3, up3], axis = 3)
    
    output1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    outputs = Conv2D(1, 1, activation = 'sigmoid')(output1)

    model = Model(inputs = inputs, outputs = outputs)

    return model

def train_model():

    model = unet()

    model.build((None, 256, 256,1))

    model.summary()

    flops = get_flops(model)
    print(f"FLOPS: {flops / 10 ** 9:.05} G")
    
    data, data_noise = input_data_train_Unet()

    validation_data, validation_data_noise = input_data_validation_Unet()

    optimizer = Adam(0.0001)

    model.compile(
                        loss=['mae'],
                        optimizer = optimizer,
                        metrics = ['mse','mae']
                        )
    
    reduce_lr =CosineAnnealingScheduler(T_max=200, eta_max=1e-4, eta_min=1e-6)
    
    model.fit(x=data_noise, y=data, validation_data=(validation_data_noise, validation_data), batch_size=8, epochs=200, verbose=1, callbacks=[reduce_lr])
    
    model.save('/home/sp432cy/sp432cy/Final_Model/Attention_Unet', save_format='tf')

if __name__ == '__main__':
    train_model()
