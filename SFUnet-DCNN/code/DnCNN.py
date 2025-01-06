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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from keras_flops import get_flops


def DnCNN():

    input = Input(shape=(256, 256, 1),name='input')
    x = Conv2D(64, kernel_size= (3,3), padding='same',name='conv2d_l1')(input)
    x = Activation('relu',name='act_l1')(x)
    for i in range(17):
        x = Conv2D(64, kernel_size=(3,3), padding='same',name='conv2d_'+str(i))(x)
        x = BatchNormalization(axis=-1,name='BN_'+str(i))(x)
        x = Activation('relu',name='act_'+str(i))(x)   
    x = Conv2D(1, kernel_size=(3,3), padding='same',name='conv2d_l3')(x)
    x = Subtract(name='subtract')([input, x])   
    model = Model(input,x)
    
    return model

def train_model():

    model = DnCNN()

    model.build((1, 256, 256,1))

    model.summary()

    flops = get_flops(model,batch_size=1)
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
    
    model.fit(x=data_noise, y=data, validation_data=(validation_data_noise , validation_data), batch_size=8, epochs=200, verbose=1, callbacks=[reduce_lr])
    
    model.save('/home/sp432cy/sp432cy/Swin-Unet/Final_Model/DnCNN', save_format='tf')

if __name__ == '__main__':
    train_model()
