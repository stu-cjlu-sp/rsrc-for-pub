from keras import *
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.activations import *

class ConvBlock(Layer):
    def __init__(self,  out_channel):
        super(ConvBlock, self).__init__()
        
        self.out_c =  out_channel   

        self.conv_block = Sequential([
            ZeroPadding2D(padding=((1, 1), (0, 0))), 
            Conv2D(self.out_c, kernel_size=(3, 1), padding='valid', use_bias=False),
            Activation('relu'),
            BatchNormalization()
        ])

    def call(self, x,):
        return self.conv_block(x)

class MultiScaleModule(Layer):
    def __init__(self, out_channel):
        super(MultiScaleModule, self).__init__()
        self.out_c = out_channel

        self.conv_3 = Sequential([
            ZeroPadding2D(padding=((1, 1), (0, 0))), 
            Conv2D(self.out_c // 3, kernel_size=(3, 2), padding='valid', use_bias=False),
            Activation('relu'),
            BatchNormalization()
        ])

        self.conv_5 = Sequential([
            ZeroPadding2D(padding=((2, 2), (0, 0))), 
            Conv2D(self.out_c // 3, kernel_size=(5, 2), padding='valid', use_bias=False),
            Activation('relu'),
            BatchNormalization()
        ])

        self.conv_7 =  Sequential([
            ZeroPadding2D(padding=((3, 3), (0, 0))),  
            Conv2D(self.out_c // 3, kernel_size=(7, 2), padding='valid', use_bias=False),
            Activation('relu'),
            BatchNormalization()
        ])

    def call(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        x = Concatenate(axis=-1)([y1, y2, y3])
        return x


class TinyMLP(Layer):
    def __init__(self, N):
        super(TinyMLP, self).__init__()
        self.N = N

        self.mlp = Sequential([
            Dense(48),  
            Activation('relu'),
            Dense(self.N),  
            Activation('tanh')  
        ])

    def call(self, x):
        return self.mlp(x)


class AdaptiveCorrectionModule(Layer):
    def __init__(self, N):
        super(AdaptiveCorrectionModule, self).__init__()
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

    def call(self, x):

        x_init = tf.identity(x)  
        x_fft = tf.signal.fft(tf.cast(x, tf.complex64))  
        X_re = tf.math.real(x_fft)
        X_im = tf.math.imag(x_fft)
        h_re = self.Re(X_re)
        h_im = self.Im(X_im)
        x_complex = tf.complex(h_re * X_re, h_im * X_im)
        x_ifft = tf.signal.ifft(x_complex)
        x_real = tf.math.real(x_ifft)
        x_out = x_real + x_init
        return x_out


class FeatureFusionModule(Layer):
    def __init__(self, num_attention_heads, hidden_size):
        super(FeatureFusionModule, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.key_layer = Dense(hidden_size, use_bias=False)
        self.query_layer = Dense(hidden_size, use_bias=False)
        self.value_layer = Dense(hidden_size, use_bias=False)

        self.dropout_attn = Dropout(0.5)
        self.dropout_v = Dropout(0.5)
        self.proj_v = Dense(hidden_size, use_bias=False)

    def trans_to_multiple_heads(self, x):
        B, N, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        new_shape = (B, N, self.num_attention_heads, self.attention_head_size)
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=False):

        query = self.query_layer(x)  
        key = self.key_layer(x)  
        value = self.value_layer(x)  

        query_heads = self.trans_to_multiple_heads(query) 
        key_heads = self.trans_to_multiple_heads(key)  
        value_heads = self.trans_to_multiple_heads(value)  

        attention_scores = tf.matmul(query_heads, key_heads, transpose_b=True)  
        attention_scores = attention_scores / tf.math.sqrt(float(self.attention_head_size))

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)  
        attention_probs = self.dropout_attn(attention_probs, training=training)

        context = tf.matmul(attention_probs, value_heads) 

        B, _, _, N  = tf.shape(context)[0], tf.shape(context)[1], tf.shape(context)[2],tf.shape(context)[3]

        context = tf.reshape(tf.transpose(context, perm=[0, 2, 1, 3]), (B, -1, N))

        x_out = self.dropout_v(self.proj_v(context), training=training)

        return x_out
    
class AMCNet(Model):
    def __init__(self,
                 num_classes=11,
                 sig_len=128,
                 extend_channel=36,
                 latent_dim=512,
                 num_heads=2,
                 conv_chan_list=None):
        
        super(AMCNet, self).__init__()
        self.sig_len = sig_len
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = conv_chan_list or [36, 64, 128, 256]

        self.ACM = AdaptiveCorrectionModule(self.sig_len)
        self.MSM = MultiScaleModule(self.extend_channel)

        self.Conv_stem = Sequential()
        for t in range(len(self.conv_chan_list) - 1):
            self.Conv_stem.add(ConvBlock(self.conv_chan_list[t+1]))

        self.FFM = FeatureFusionModule(self.num_heads, self.sig_len) 

        self.GAP = GlobalAveragePooling1D()

        self.classifier = Sequential([
            Dense(512),
            Dropout(0.5),   
            ReLU(),
            Dense(self.num_classes ,activation="softmax")
        ])

    def call(self, x):
        x = tf.expand_dims(x, axis=-1) 
        x = tf.transpose(x, perm=[0, 3, 2, 1])
        x = self.ACM(x)
        x = tf.transpose(x, perm=[0, 3, 2 ,1])
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = tf.squeeze(x, axis=2) 
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.FFM(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.GAP(x)
        y = self.classifier(x)
        return y
    
if __name__ == '__main__':

    model = AMCNet()
    test_input = tf.random.normal((1, 128, 2))
    output = model(test_input)
    print(output)