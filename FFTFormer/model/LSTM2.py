from keras import *
from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf

class LSTM2(Model):
    def __init__(self,  **kwargs):
        super(LSTM2, self).__init__(**kwargs)
        self.lstm1 = LSTM(units=128, return_sequences=True)
        self.lstm2 = LSTM(units=128)
        self.fc = Dense(11, activation="softmax")

    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.fc(x)
        return x

