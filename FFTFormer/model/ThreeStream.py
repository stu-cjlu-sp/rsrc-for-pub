from keras import *
from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf


class ThreeStream(Model):
    def __init__(self, classes=11):
        super(ThreeStream, self).__init__()

        self.conv1 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
            MaxPool1D(pool_size=2, strides=1, padding="same"),
        ])
        self.conv2 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
            MaxPool1D(pool_size=2, strides=1, padding="same"),
        ])

        self.conv3 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
            MaxPool1D(pool_size=2, strides=1, padding="same"),
        ])

        self.conv4 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
            MaxPool1D(pool_size=2, strides=1, padding="same"),
        ])

        self.conv5 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
            MaxPool1D(pool_size=2, strides=1, padding="same"),
        ])

        self.conv6 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
        ])

        self.conv7 = Sequential([
            Conv1D(64, 3, padding ="same"),
            ReLU(),
            MaxPool1D(pool_size=2, strides=1, padding="same"),
        ])

        self.lstm1 = LSTM(64, return_sequences=True)
        self.lstm2 = LSTM(64)

        self.fc1 = Sequential([
            Dense(64),
            ReLU()
        ])
        self.fc2 = Dense(classes, activation='softmax')

    def call(self, x, training=None):

        B = tf.shape(x)[0]
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        A = tf.reshape(x1, (B, -1, 1))
        P = tf.reshape(x2, (B, -1, 1))
        F = tf.reshape(x3, (B, -1, 1))

        Conv1 =self.conv1(A)
        Conv3 =self.conv3(P)
        Conv5 =self.conv5(F)
        Conv2 =self.conv2(Conv1)
        Conv4 =self.conv4(Conv3)
        Conv6 =self.conv6(Conv5)

        Concat1= Concatenate(axis=-1)([Conv2, Conv4])
        Conv7 = self.conv7(Concat1)
        Concat2= Concatenate(axis=-1)([Conv6, Conv7])
        N = tf.shape(Concat2)[1]
        Reshape1 = tf.reshape(Concat2, (B, N*2,-1))

        LSTM1 = self.lstm1(Reshape1)
        LSTM2 = self.lstm2(LSTM1)
        FC1 = self.fc1(LSTM2)
        FC2 = self.fc2(FC1)

        return FC2
    
if __name__ == "__main__":
    x = tf.random.normal((1, 128, 3))
    model = ThreeStream(11)
    output = model(x)
    print(output.shape) 