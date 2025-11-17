import tensorflow as tf
from keras.models import *
from keras.layers import *
from sklearn.metrics import *

class MCLDNN(Model):
    def __init__(self, classes=11, signal_length=128, **kwargs):
        super(MCLDNN, self).__init__(**kwargs)

        self.signal_length = signal_length

        self.conv1 = Conv2D(50, (8,2), padding='same', activation="relu")
        self.conv2 = Conv1D(50, 8, padding='causal', activation="relu")
        self.conv3 = Conv1D(50, 8, padding='causal', activation="relu")
        self.conv4 = Conv2D(50, (8,1), padding='same', activation="relu")
        self.conv5 = Conv2D(100, (5,2), padding='valid', activation="relu")
        self.lstm1 = LSTM(128, return_sequences=True)
        self.lstm2 = LSTM(128)

        self.fc1 = Sequential([
            Dense(128),
            Activation('selu'),
            Dropout(0.5),])
        self.fc2 = Sequential([
            Dense(128),
            Activation('selu'),
            Dropout(0.5), ])
        self.fc3 = Sequential([
            Dense(classes),
            Activation('softmax'),])

    def call(self, x):
        
        N=self.signal_length
        IQ = x
        I = x[:, :, 0:1]
        Q = x[:, :, 1:2]
        IQ = Reshape((N, -1,  1))(IQ)

        Conv1 = self.conv1(IQ)
        Conv2 = self.conv2(I)
        Conv3 = self.conv3(Q)
        Conv2_reshaped = Reshape((N, 1, -1))(Conv2)
        Conv3_reshaped = Reshape((N, 1, -1))(Conv3)
        Concat1= Concatenate(axis=2)([Conv2_reshaped, Conv3_reshaped])
        Conv4 = self.conv4(Concat1)
        Concat2= Concatenate(axis=-1)([Conv1, Conv4])
        Conv5 = self.conv5(Concat2)
        Conv5_reshaped = Reshape((-1, 100))(Conv5)

        LSTM1 = self.lstm1(Conv5_reshaped)
        LSTM2 = self.lstm2(LSTM1)
        FC1 = self.fc1(LSTM2)
        FC2 = self.fc2(FC1)
        FC3 = self.fc3(FC2)

        return FC3
    
if __name__ == "__main__":
    x = tf.random.normal((1, 128, 2))
    model = MCLDNN(11)
    output = model(x)
    print(output.shape)  