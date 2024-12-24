import tensorflow as tf
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import h5py
import os
from resnet_ca import ResNet_ca
from aff_resnet import ResNet_aff 
from loss import nt_xent

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:       
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

classes_name = ['LFM', 'Costas', 'BPSK', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
train_IQ = np.load('/Data/300/Data_IQ_1.npy')
train_labels = np.load('/Data/300/label_1.npy')
train_LFM1 = h5py.File('/Data/300/data_cwd_2.mat')
train_LFM1 = train_LFM1['yt_out']
train_LFM1 = np.transpose(train_LFM1,(2,0,1))
train_LFM1 = train_LFM1[:, :, :, np.newaxis]
idx = np.random.permutation(train_IQ.shape[0])
train_data, train_LFM, train_label = train_IQ[idx], train_LFM1[idx], train_labels[idx]

BATCH_SIZE = 32
train_ds = tf.data.Dataset.from_tensor_slices((train_IQ, train_LFM1, train_labels)).batch(BATCH_SIZE)

def model_projection(h, hidden_1, hidden_2, hidden_3):
    projection_1 = Dense(hidden_1)(h)
    projection_1 = BatchNormalization()(projection_1)
    projection_1 = Activation("relu")(projection_1)
    
    projection_2 = Dense(hidden_2)(projection_1)
    z = BatchNormalization()(projection_2)
    return Model(inputs=h, outputs=z)

def get_resnet_simclr(hidden_1, hidden_2, hidden_3,input_shape1=(2,1024,1), input_shape2=(64,64,1)):
    inputs1 = Input(input_shape1)
    inputs2 = Input(input_shape2)
    base_model1 = ResNet_aff(im_height=input_shape1[0], im_width=input_shape1[1])
    base_model2 = ResNet_ca(im_height=input_shape2[0], im_width=input_shape2[1])
    h1 = base_model1(inputs1)
    h2 = base_model2(inputs2)
    projection_model1 = model_projection(h1, hidden_1, hidden_2, hidden_3)
    projection_model2 = model_projection(h2, hidden_1, hidden_2, hidden_3)
    z1 = projection_model1(h1)
    z2 = projection_model2(h2)
    base_model11 = Model(inputs=inputs1, outputs=h1)
    base_model22 = Model(inputs=inputs2, outputs=h2)
    resnet_simclr = Model(inputs=[inputs1, inputs2], outputs=[h1, h2, z1, z2])
    return resnet_simclr,base_model11, base_model22

@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        h1, h2, z1, z2 = model([xis, xjs])
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)
        loss1 = nt_xent(z1, z2, BATCH_SIZE, temperature)
        reg_loss = tf.add_n(model.losses) if model.losses else 0
        loss = loss1 +  reg_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, loss1, reg_loss

def train_simclr(model, dataset, optimizer, criterion, epochs,
                 temperature=0.1):
    step_wise_loss = []
    epoch_wise_loss = []
    step_wise_loss1 = []
    step_wise_reg_loss = []
    best_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        for image_batch, imf_batch, label_batch in dataset:
            a = image_batch
            a = a[:, :, :,np.newaxis]
            b = imf_batch 
            loss, loss1, reg_loss= train_step(a, b, model, optimizer, criterion, temperature)
            step_wise_loss.append(loss)
            step_wise_loss1.append(loss1)
            step_wise_reg_loss.append(reg_loss)
        epoch_loss = np.mean(step_wise_loss)
        epoch_wise_loss.append(epoch_loss)
        if epoch_loss < best_loss:
            tf.keras.models.save_model(model, '/Pretrained_Weights/base_mode_resnet18_AFF_coordatt.h5', overwrite=True)
            best_loss = epoch_loss
        if epoch % 1 == 0:
            print("epoch: {} loss: {:.10f}".format(epoch + 1, epoch_loss))
    return epoch_wise_loss, model

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)

steps_per_epoch= train_data.shape[0] // BATCH_SIZE
initial_learning_rate = 0.001
decay_steps = 100 * steps_per_epoch
decay_rate = 0.1
print("steps_per_epoch:", steps_per_epoch)
print("initial_learning_rate:", initial_learning_rate)
print("decay_steps:", decay_steps)
print("decay_rate:", decay_rate)
lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
resnet_simclr_2,base_model11, base_model22 = get_resnet_simclr(256,512,256) 
base_model22.summary()
epoch_wise_loss, resnet_simclr = train_simclr(resnet_simclr_2, train_ds, optimizer, criterion, epochs=300, temperature=0.1)

plt.plot(epoch_wise_loss)
plt.show()
plt.savefig('/Pretrained_Weights/epoch_wise_mode_resnet18_AFF_coordatt.jpg', bbox_inches = 'tight')
filename1 = "/Pretrained_Weights/base_mode_resnet18_AFF_coordatt.h5"
filename2 = "/Pretrained_Weights/weights_mode_resnet18_AFF_coordatt.h5"
base_model11.save_weights(filename1)
resnet_simclr_2.save_weights(filename2)