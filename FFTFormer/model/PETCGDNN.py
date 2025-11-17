import tensorflow as tf
from keras.models import *
from keras.layers import *
from sklearn.metrics import *

class PETCGDNN(Model):
    def __init__(self, classes=11, **kwargs):
        super(PETCGDNN, self).__init__(**kwargs)
        
        self.dr = 0.5 
        
        self.flatten = Flatten()
        self.dense = Dense(1, name='fc1')
        self.activation_linear = Activation('linear')

        self.cos = Lambda(lambda x: tf.keras.backend.cos(x))
        self.sin = Lambda(lambda x: tf.keras.backend.sin(x))

        self.multiply_i_cos = Multiply()
        self.multiply_q_sin = Multiply()
        self.multiply_q_cos = Multiply()
        self.multiply_i_sin = Multiply()

        self.add_y1 = Add()
        self.subtract_y2 = Subtract()

        self.reshape_y1 = Reshape(target_shape=(128, 1), name='reshape1')
        self.reshape_y2 = Reshape(target_shape=(128, 1), name='reshape2')
        self.reshape_x3 = Reshape(target_shape=((128, 2, 1)), name='reshape3')

        self.conv1 = Conv2D(75, (8,2), padding='valid', activation="relu", name="conv1")
        self.conv2 = Conv2D(25, (5,1), padding='valid', activation="relu", name="conv2")

        self.reshape4 = Reshape(target_shape=((117,25)), name='reshape4')
        self.gru = GRU(units=128)

        self.fc = Sequential([
            Dense(classes, activation='softmax', name='softmax'),
        ])

    def call(self, x):

        IQ = x
        I = x[:, :, 0]
        Q = x[:, :, 1]
        
        x1 = self.flatten(IQ)
        x1 = self.dense(x1)
        x1 = self.activation_linear(x1)

        cos1 = self.cos(x1)
        sin1 = self.sin(x1)

        x11 = self.multiply_i_cos([I, cos1])
        x12 = self.multiply_q_sin([Q, sin1])

        x21 = self.multiply_q_cos([Q, cos1])
        x22 = self.multiply_i_sin([I, sin1])

        y1 = self.add_y1([x11,x12])
        y2 = self.subtract_y2([x21,x22])

        y1 = self.reshape_y1(y1)
        y2 = self.reshape_y2(y2)

        x11 = Concatenate()([y1, y2])
        x3 = self.reshape_x3(x11)

        x3 = self.conv1(x3)
        x3 = self.conv2(x3)

        x4 = self.reshape4(x3)
        x4 = self.gru(x4)

        output = self.fc(x4)

        return output
    
if __name__ == "__main__":
    x = tf.random.normal((1, 128, 2))
    model = PETCGDNN(11)
    output = model(x)
