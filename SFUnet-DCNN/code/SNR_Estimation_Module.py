from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam,SGD
from cosine_annealing import CosineAnnealingScheduler
from tensorflow import *
from keras.models import *
from keras.layers import *
from load_dataset import *
from keras_flops import get_flops
import datetime
from DCNN import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
config=tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
starttime = datetime.datetime.now()

def snr_estimation_module():
    
    inputs = Input(shape=(8, 8, 96))

    x = layers.GlobalAveragePooling2D()(inputs)
     
    # fc1
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # fc2
    x = layers.Dense(2)(x)

    outputs = layers.Softmax(name="SNR_Predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name='snr_level')

    return model

def Final_Model():

    inputs = Input(shape=(256, 256, 1), name='input_image1')

    x = DCNN_SFB(input_shape=(256, 256, 1), num_classes=12, alpha=1.0, include_top=False)(inputs)
    
    x = layers.GlobalAveragePooling2D()(x)
     
    # fc1
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # fc2
    x = layers.Dense(2)(x)

    outputs = layers.Softmax(name="Predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name='SNR_estimation')

    return model


def train_model():

    model = snr_estimation_module()
    model.build(input_shape=(None, 8, 8, 96)) 
    print(model.summary())
    print("Compiling Model...")

    flops = get_flops(model)
    print(f"FLOPS: {flops / 10 ** 9:.05} G")

    optimizer = Adam(0.0001)

    reduce_lr =CosineAnnealingScheduler(T_max=100, eta_max=1e-4, eta_min=1e-6)

    checkpoint_filepath = '/home/sp432cy/sp432cy/Swin-Unet/Final_Model/SNR_estimation'

    checkpoint  = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    model.compile(optimizer=optimizer,
                loss= ['CategoricalCrossentropy'],
                metrics=['accuracy'])

    data, label = input_data_train_SNR()
    validation_data, validation_label = input_data_validation_SNR()

    print(data.shape)
    print(label.shape)

    label = keras.utils.to_categorical(label, 2)
    validation_label = keras.utils.to_categorical(validation_label, 2)

    model.fit(x=data, y=label, batch_size=32, epochs=100, shuffle=True, validation_data=(validation_data,validation_label), callbacks=[reduce_lr, checkpoint])

    model.save('/home/sp432cy/sp432cy/Swin-Unet/Final_Model/SNR_estimation_final', save_format='tf')

    endtime = datetime.datetime.now()
    print('Model training time:', (endtime - starttime).seconds, 's')

def test_model():

    model = load_model('/home/sp432cy/sp432cy/Swin-Unet/Final_Model/SNR_estimation_final')

    for snr in range(-16,12, 2):

        data_noise, labels = input_data_test_SNR(snr)

        predictions = model.predict(data_noise, batch_size=1)

        predicted_labels = np.argmax(predictions, axis=1)

        # true_labels = np.argmax(labels, axis=1)

        accuracy = accuracy_score(labels, predicted_labels)

        print(f"Predicted labels at {snr} dB SNR:", predicted_labels)
        # print(f"True labels at {snr} dB SNR:", labels)
        print(f"Accuracy at {snr} dB SNR: {accuracy*100:.4f}%")

if __name__ == '__main__':
    train_model()
    test_model()