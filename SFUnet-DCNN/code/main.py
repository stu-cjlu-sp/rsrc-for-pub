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
from Clasaification_Network import *
from SNR_Estimation_Module import *
from keras_flops import get_flops
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'
config=tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
starttime = datetime.datetime.now()

class HighSNRBranch(Layer):
    def __init__(self, **kwargs):
        super(HighSNRBranch, self).__init__(**kwargs)
        self.racn = RACN(filter_num=96, reduction_ratio=8, kernel_size=7, name='RACN_high')

    def call(self, features, **kwargs):
        return self.racn(features)


class LowSNRBranch(Layer):
    def __init__(self, denoising_model, **kwargs):
        super(LowSNRBranch, self).__init__(**kwargs)
        self.denoising_model = denoising_model
        self.dcnn_sfb = DCNN_SFB(input_shape=(256, 256, 1), num_classes=12, alpha=1.0, include_top=False)
        self.racn = RACN(filter_num=96, reduction_ratio=8, kernel_size=7, name='RACN_low')

    def call(self, raw_inputs, **kwargs):
        denoised_inputs = self.denoising_model(raw_inputs)
        refined_features = self.dcnn_sfb(denoised_inputs)
        return self.racn(refined_features)



class SNRBasedRouting(Layer):
    def __init__(self, high_snr_branch, low_snr_branch, **kwargs):
        super(SNRBasedRouting, self).__init__(**kwargs)
        self.high_snr_branch = high_snr_branch
        self.low_snr_branch = low_snr_branch

    def call(self, inputs, training=None):
        raw_inputs, features, snr_level = inputs

        # Determine the SNR level (0 for high SNR, 1 for low SNR)
        snr_pred = tf.argmax(snr_level, axis=-1)  # Shape: (batch_size,)
        snr_pred = tf.cast(snr_pred, tf.int32)

        # Route to the appropriate branch
        high_snr_output = self.high_snr_branch(features)
        low_snr_output = self.low_snr_branch(raw_inputs)

        # Select outputs based on SNR prediction
        outputs = tf.where(
            tf.equal(snr_pred, 0)[:, None],  # Broadcast shape
            high_snr_output,
            low_snr_output
        )
        return outputs

def Final_Model():

    denoising_model = load_model('/home/sp432cy/sp432cy/Swin-Unet/Final_Model/SFUnet')
    denoising_model.trainable = False

    inputs = Input(shape=(256, 256, 1))

    dcnn_feature = DCNN_SFB(input_shape=(256, 256, 1), num_classes=12, alpha=1.0, include_top=False)(inputs)

    snr_level = snr_estimation_module()(dcnn_feature)

    high_snr_branch = HighSNRBranch(name="high_snr_branch")
    low_snr_branch = LowSNRBranch(denoising_model=denoising_model, name="low_snr_branch")

    classification_output = SNRBasedRouting(
        high_snr_branch=high_snr_branch,
        low_snr_branch=low_snr_branch,
        name="classification_output"
    )([inputs, dcnn_feature, snr_level])

    model = Model(inputs=inputs, outputs=[snr_level, classification_output],)

    return model

def train_model():

    model = Final_Model()
    model.build(input_shape=(None, 256, 256, 1)) 
    print(model.summary())
    print("Compiling Model...")

    # flops = get_flops(model)
    # print(f"FLOPS: {flops / 10 ** 9:.05} G")

    optimizer = Adam(0.0001)

    reduce_lr =CosineAnnealingScheduler(T_max=100, eta_max=1e-4, eta_min=1e-6)

    checkpoint_filepath = '/home/sp432cy/sp432cy/Swin-Unet/Final_Model/final_model'

    checkpoint  = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    model.compile(
        optimizer=optimizer,
        loss=['CategoricalCrossentropy', 'CategoricalCrossentropy'],
        metrics=['accuracy']
    )

    data, label1, label2 = input_data_train_Final_Model()
    validation_data, validation_label1, validation_label2 = input_data_validation_Final_Model()

    label1 = keras.utils.to_categorical(label1, 2)
    label2 = keras.utils.to_categorical(label2, 12)

    validation_label1 = keras.utils.to_categorical(validation_label1, 2)
    validation_label2 = keras.utils.to_categorical(validation_label2, 12)

    print(data.shape)
    print(label1.shape)
    print(label2.shape)

    model.fit(x=data, y=[label1, label2], batch_size=32, epochs=100, shuffle=True, validation_data=(validation_data, [validation_label1,validation_label2]), callbacks=[reduce_lr,checkpoint])

    model.save('/home/sp432cy/sp432cy/Swin-Unet/Final_Model/final_model', save_format='tf')

    endtime = datetime.datetime.now()
    print('Model training time:', (endtime - starttime).seconds, 's')

def test_model():

    model = load_model('/home/sp432cy/sp432cy/Swin-Unet/Final_Model/final_model')

    for snr in range(-16, 12, 2):

        data_noise, label1, label2 = input_data_test_Final_Model(snr)

        predictions = model.predict(data_noise, batch_size=1)

        snr_predicted_labels = np.argmax(predictions[0], axis=1)

        predicted_labels = np.argmax(predictions[1], axis=1)

        # true_labels = np.argmax(labels, axis=1)
        snr_accuracy = accuracy_score(label1, snr_predicted_labels)

        accuracy = accuracy_score(label2, predicted_labels)

        predicted_labels_str = ', '.join(map(str, predicted_labels))
        
        print(f"Predicted labels at {snr} dB SNR: [{predicted_labels_str}]")
        # print(f"True labels at {snr} dB SNR:", labels)
        print(f"SNR Accuracy at {snr} dB SNR: {snr_accuracy*100:.4f}%")

        print(f"Accuracy at {snr} dB SNR: {accuracy*100:.4f}%")

if __name__ == '__main__':

    train_model()
    test_model()