
from keras.callbacks import ModelCheckpoint
from cosine_annealing import CosineAnnealingScheduler
import numpy as np
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from tensorflow import *
from keras.models import *
from keras.layers import *
from load_dataset import *
from DCNN import *
from AMCN import *
from keras_flops import get_flops
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
starttime = datetime.datetime.now()


def _make_divisible(ch, divisor=8, min_ch=None):

    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)

def Final_Model():

    base_model1 = load_model('/home/sp432cy/sp432cy/Final_Model/SwinUnet_final')
    base_model1.trainable = False
    
    inputs = Input(shape=(256, 256, 1), name='input_image1')

    x = base_model1(inputs)

    x = DCNN(input_shape=(256, 256, 1), num_classes=12, alpha=1.0, include_top=False)(x)

    x = CBAM_Block(input_layer=x, filter_num=96, reduction_ratio=8, kernel_size=7, name='Attention_Mechanism_Classifier')
    
    last_c = _make_divisible(48 * 6 * 1)

    x = layers.Conv2D(filters=last_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name="Conv_1")(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwish(name="Conv_1/HardSwish")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 288))(x)
     
    # fc1
    x = layers.Conv2D(filters=128,
                        kernel_size=1,
                        padding='same',
                        name="fc_1")(x)
    x = HardSwish(name="fc_1/HardSwish")(x)

    # fc2
    x = layers.Conv2D(filters=12,
                        kernel_size=1,
                        padding='same',
                        name="fc_2")(x)
    x = layers.Flatten()(x)
    outputs = layers.Softmax(name="Predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name='SwinUnet_DCNN')

    return model

def train_model():

    model = Final_Model()
    model.build(input_shape=(None, 256, 256, 1)) 
    print(model.summary())
    print("Compiling Model...")

    flops = get_flops(model)
    print(f"FLOPS: {flops / 10 ** 9:.05} G")

    optimizer = Adam(0.0001)

    reduce_lr =CosineAnnealingScheduler(T_max=100, eta_max=1e-4, eta_min=1e-6)

    checkpoint_filepath = '/home/sp432cy/sp432cy/Final_Model/SwinUnet_DCNN'

    checkpoint  = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    model.compile(optimizer=optimizer,
                loss= ['CategoricalCrossentropy'],
                metrics=['accuracy'])

    data, label = input_data_train_DCNN()
    validation_data, validation_label = input_data_validation_DCNN()

    print(data.shape)
    print(label.shape)

    label = keras.utils.to_categorical(label, 12)
    validation_label = keras.utils.to_categorical(validation_label, 12)

    model.fit(x=data, y=label, batch_size=32, epochs=100, shuffle=True, validation_data=(validation_data,validation_label), callbacks=[reduce_lr,checkpoint])

    model.save('/home/sp432cy/sp432cy/Final_Model/SwinUnet_DCNN_final', save_format='tf')

    endtime = datetime.datetime.now()
    print('Model training time:', (endtime - starttime).seconds, 's')

def test_model():

    model = load_model('/home/sp432cy/sp432cy/Final_Model/Without_Unet_final')

    for snr in range(-14,12, 2):

        data_noise, labels = input_data_test_DCNN(snr)

        predictions = model.predict(data_noise, batch_size=1)

        predicted_labels = np.argmax(predictions, axis=1)

        # true_labels = np.argmax(labels, axis=1)

        accuracy = accuracy_score(labels, predicted_labels)

        predicted_labels_str = ', '.join(map(str, predicted_labels))
        
        print(f"Predicted labels at {snr} dB SNR: [{predicted_labels_str}]")
        # print(f"True labels at {snr} dB SNR:", labels)
        print(f"Accuracy at {snr} dB SNR: {accuracy*100:.4f}%")

if __name__ == '__main__':

    train_model()
    test_model()