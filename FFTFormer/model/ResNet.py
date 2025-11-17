from keras import *
from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf

class ResidualStack(Layer):
    def __init__(self, channel, kernelsize, poolsize, stride, padding, **kwargs):
        super(ResidualStack, self).__init__(**kwargs)
        self.conv1 = Conv1D(channel, kernel_size=1, use_bias=False)
        
        self.unit1 = Sequential([
            Conv1D(channel, kernel_size=kernelsize, strides=stride, padding=padding, use_bias=False),
            ReLU(),
            Conv1D(channel, kernel_size=kernelsize, strides=stride, padding=padding, use_bias=False)
        ])
        
        self.unit2 = Sequential([
            Conv1D(channel, kernel_size=kernelsize, strides=stride, padding=padding, use_bias=False),
            ReLU(),
            Conv1D(channel, kernel_size=kernelsize, strides=stride, padding=padding, use_bias=False)
        ])
        
        self.relu = ReLU()
        self.pool = MaxPooling1D(poolsize, strides=poolsize)

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.unit1(x1) + x1
        x3 = self.relu(x2)
        x4 = self.unit2(x3) + x3
        x5 = self.relu(x4)
        output = self.pool(x5)
        return output
    
class ResNet(Model):
    def __init__(self, classes=11, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.ReStk0 = self.get_layer(32, 3, 2, 1, 'same')
        self.ReStk1 = self.get_layer(32, 3, 2, 1, 'same')
        self.ReStk2 = self.get_layer(32, 3, 2, 1, 'same')
        self.ReStk3 = self.get_layer(32, 3, 2, 1, 'same')
        self.ReStk4 = self.get_layer(32, 3, 2, 1, 'same')
        self.ReStk5 = self.get_layer(32, 3, 2, 1, 'same')

        self.flat = Flatten()
        self.fc3 = Sequential([Dense(128), 
                                    Activation('selu'), 
                                    Dropout(0.3)])
        self.fc4 = Sequential([Dense(128), 
                                    Activation('selu'), 
                                    Dropout(0.3)])
        self.fc5 = Dense(classes,activation='softmax')
        
        self.init_weights()

    def init_weights(self):
        initializer = tf.keras.initializers.GlorotUniform()
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = initializer
            if hasattr(layer, 'bias_initializer') and layer.use_bias is not None:
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def get_layer(self, channel, kernelsize, poolsize, stride, padding):
        return ResidualStack(channel, kernelsize, poolsize, stride, padding)

    def call(self, x):

        x = self.ReStk0(x)
        x = self.ReStk1(x)
        x = self.ReStk2(x)
        x = self.ReStk3(x)
        x = self.ReStk4(x)
        x = self.ReStk5(x)

        x = self.flat(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

if __name__ == '__main__':

    model = ResNet()
    test_input = tf.random.normal((1, 128, 2))
    output = model(test_input)
    print(output)