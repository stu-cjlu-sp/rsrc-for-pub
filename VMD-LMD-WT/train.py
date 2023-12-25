import numpy as np
import keras
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import optimizers
import keras.backend as K
import CNN as cnn
import cv2, os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


# Function to load training data and labels
def load_data_train(directory_name):
    train_image0 = []
    train_label0 = []
    
    # Iterate over subdirectories (each subdirectory represents a class)
    i = -1
    for last_file in os.listdir(directory_name + '/'):
        i = i + 1
        for filename in os.listdir(directory_name + '/' + last_file):
            img = cv2.imread(directory_name + '/' + last_file + '/' + filename, cv2.IMREAD_GRAYSCALE)
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = img / 255
            train_image0.append(img)
            train_label0.append(i)

    train_image2,  train_label2 = np.stack(train_image0),  np.array(train_label0)
    idx = np.random.permutation(train_image2.shape[0])
    train_image2, train_label2 = train_image2[idx], train_label2[idx]
    train_image2 = np.array(train_image2)
    train_label2 = np.array(train_label2)
    train_image2 = train_image2[:, :, :, np.newaxis]
    return train_image2, train_label2



# Learning rate scheduler
def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

# Create a LearningRateScheduler callback
reduce_lr = LearningRateScheduler(scheduler)

# Load training data and labels
directory_name = 'EMD/train'
train_image2, train_label2 = load_data_train(directory_name=directory_name)
train_data, test_data, train_label, test_label = train_test_split(train_image2, train_label2, test_size=0.20,
                                                                  random_state=42)
# Convert labels to one-hot encoding
train_label = keras.utils.to_categorical(train_label, 12)
test_label = keras.utils.to_categorical(test_label, 12)

# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 64    # training batch size
adam0 = optimizers.Adam(lr=0.001)

# Build framework
model=cnn.CNN()
model.summary()
filepath = 'EMD/vmd-lmd-1.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
model.compile(loss='categorical_crossentropy', optimizer=adam0, metrics=['accuracy'])

# Train the framework
model.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True,
          validation_data=(test_data, test_label), callbacks=[reduce_lr, checkpoint])
