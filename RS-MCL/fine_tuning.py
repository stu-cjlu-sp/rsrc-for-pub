import tensorflow as tf
from keras.layers import *
from keras.models import *
import tensorflow as tf
import numpy as np
import os
import h5py
import keras
from keras.optimizers import Adam
from resnet_ca import ResNet_ca
from aff_resnet import ResNet_aff
import random
import tqdm

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

classes_name = ['LFM', 'Costas', 'BPSK', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']

def select_random_samples(root_folder, num_samples_per_folder=54):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    selected_samples = []
    for folder in subfolders:
        files = [f.name for f in os.scandir(folder) if f.is_file()]
        selected_files = random.sample(files, min(num_samples_per_folder, len(files)))
        selected_samples.extend([os.path.join(folder, file) for file in selected_files])
    return selected_samples

def prepare_images(image_paths):
    labels = []
    train_datas = []
    for image in image_paths:
        mat_file = h5py.File(image)
        data_key = list(mat_file)
        data = mat_file[data_key[0]]
        if data.shape[1]>=1024:
            data = data[:, :1024] 
        else: 
            a = 1024-data.shape[1]
            data =np.concatenate((data, np.zeros((data.shape[0], a))), axis=1)
        train_datas.append(data)
        label = image.split("/")[3]
        label_index = classes_name.index(label)
        labels.append(label_index)
    train_datas = np.array(train_datas)
    labels = np.array(labels)
    return train_datas, labels

def train_test_split(data1,label_snr, test_ratio):
    random_state = 42                                      
    np.random.seed(random_state)                           
    shuffled_indices = np.random.permutation(data1.shape[0])
    test_set_size = int(len(data1) * test_ratio)            
    test_indices = shuffled_indices[:test_set_size]         
    train_indices = shuffled_indices[test_set_size:]        
    return data1[train_indices], label_snr[train_indices], \
           data1[test_indices], label_snr[test_indices]

train_images_paths = select_random_samples("Data/300_1/fing_300")
train_IQ, train_label = prepare_images(train_images_paths)

train_IQ, train_label, test_IQ, test_label = train_test_split(train_IQ, train_label, test_ratio=0.20)
train_IQ, train_label, test_IQ, test_label = np.array(train_IQ), np.array(train_label), np.array(test_IQ), np.array(test_label)
train_IQ = train_IQ[:, :, :, np.newaxis]
test_IQ = test_IQ[:, :, :, np.newaxis]

train_label_enc = keras.utils.to_categorical(train_label, 12)
test_label_enc = keras.utils.to_categorical(test_label, 12)

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


resnet_simclr, base_model11, base_model22= get_resnet_simclr(256,512,256)

base_model11.load_weights('/Pretrained_Weights/base_mode_resnet18_AFF_coordatt.h5')
base_model11 = Model(base_model11.input, base_model11.output)
base_model11.summary()

x = Flatten()(base_model11.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model11.input, outputs=output)
model.summary()

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_IQ, train_label_enc)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_IQ, test_label_enc))

encoder_learning_rate = 5e-4
classifier_learning_rate = 1e-3
encoder = Model(inputs=base_model11.input, outputs=base_model11.output)
classifier = Model(inputs=encoder.output, outputs=output)
encoder.summary()
classifier.summary()
encoder_optimizer = Adam(learning_rate=encoder_learning_rate)
classifier_optimizer = Adam(learning_rate=classifier_learning_rate)
encoder.trainable = True
classifier.trainable = True
encoder.compile(optimizer=encoder_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.compile(optimizer=classifier_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def scheduler(epoch, lr):
    if epoch % 100 == 0 and epoch != 0:
        return lr * 0.1
    return lr


filepath = '/fine-tuning_weights/base_mode_resnet18_AFF_coordatt.h5'
best_val_accuracy = 0.0
for epoch in range(200):
    for x, y in train_dataset:
        with tf.GradientTape(persistent=True) as tape:
            encoding = encoder(x)
            predictions = classifier(encoding)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)

        gradients_encoder = tape.gradient(loss, encoder.trainable_variables)
        encoder_optimizer.apply_gradients(zip(gradients_encoder, encoder.trainable_variables))

        gradients_classifier = tape.gradient(loss, classifier.trainable_variables)
        classifier_optimizer.apply_gradients(zip(gradients_classifier, classifier.trainable_variables))
    del tape 
    reduce_lr_encoder = tf.keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr_encoder.set_model(encoder)
    reduce_lr_classifier = tf.keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr_classifier.set_model(classifier)
    loss1= tf.reduce_mean(loss)
    val_accuracy = np.mean([np.argmax(classifier(encoder(tf.expand_dims(x, 0)))) == np.argmax(y) for x, y in test_dataset])
    print(f'Epoch {epoch + 1}, Loss: {loss1.numpy()}, Validation Accuracy: {val_accuracy}')
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        combined_model = Model(inputs=encoder.input, outputs=classifier(encoder.output))
        combined_model.save(filepath)
