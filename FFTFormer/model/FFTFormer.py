from keras import *
from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf


class ConvEncoder(Layer):
    def __init__(self, dim,):
        super(ConvEncoder, self).__init__()
        self.dwconv1 = DepthwiseConv1D(kernel_size=1, padding='same')
        self.dwconv3 = DepthwiseConv1D(kernel_size=3, padding='same')
        self.dwconv5 = DepthwiseConv1D(kernel_size=5, padding='same')
        self.dwconv7 = DepthwiseConv1D(kernel_size=7, padding='same')
        self.concat = Concatenate(axis=-1)
        self.bn = BatchNormalization()
        self.swish = Activation('swish')
        self.conv = Conv1D(filters=dim, kernel_size=1,  padding='same')
        self.dropout = Dropout(rate=0.2)

    def call(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv3(x)
        x3 = self.dwconv5(x)
        x4 = self.dwconv7(x)

        x = self.concat([x1, x2, x3, x4])
        x = self.bn(x)
        x = self.swish(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class FAFFN(Layer):
    def __init__(self, dim, hidden_dim, h,w,drop=0., **kwargs):
        super().__init__(**kwargs)

        self.complex_weight  = self.add_weight(
            shape=(h, w, dim*2, 2),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name='real_weight'
        )

        self.norm1 = BatchNormalization(epsilon=1e-5)
        self.fc1 = Dense(hidden_dim,  use_bias=False)
        self.act = Activation('gelu')
        self.fc2 = Dense(dim, use_bias=False)
        self.drop = Dropout(drop)

    def call(self, x):

        x = self.norm1(x)
        x = self.fc1(x)
        x1,x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        x = self.act(x1) *x2
        x = self.drop(x)
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        a = b = tf.cast(tf.math.sqrt(tf.cast(N, tf.float32)), tf.int32)
        x_fft = tf.reshape(x, [B, a, b, C])
        x_fft = tf.cast(x_fft, tf.float32)
        x_fft = tf.transpose(x_fft, perm=[0, 3, 1, 2])  
        x_fft = tf.signal.rfft2d(x_fft) 
        x_fft = tf.transpose(x_fft, perm=[0, 2, 3, 1])  
        complex_weight = tf.complex(self.complex_weight[..., 0], self.complex_weight[..., 1])
        x_fft = x_fft * complex_weight
        x_fft = tf.transpose(x_fft, perm=[0, 3, 1, 2])
        x_fft = tf.signal.irfft2d(x_fft, fft_length=[a, b])
        x_fft = tf.transpose(x_fft, perm=[0, 2, 3, 1])  
        x_fft = tf.reshape(x_fft, [B, N, C])
        x = self.fc2(x_fft)

        return x

    
class FFTAttention(Layer):
    def __init__(self, dim, drop):
        super(FFTAttention, self).__init__()

        self.to_qkv = Dense(dim * 3, use_bias=False)
        self.to_out = Dense(dim, use_bias=False)

        self.dropout_attn = Dropout(drop)
        self.dropout_v = Dropout(drop)
        self.norm = LayerNormalization(epsilon=1e-5)

    def call(self, x):

        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        a = b = tf.cast(tf.math.sqrt(tf.cast(N, tf.float32)), tf.int32)

        qkv =self.to_qkv(x)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

        q_fft = tf.reshape(q, [B, a, b, C])
        q_fft = tf.cast(q_fft, tf.float32)
        q_fft = tf.transpose(q_fft, perm=[0, 3, 1, 2])  
        q_fft = tf.signal.rfft2d(q_fft) 

        k_fft = tf.reshape(k, [B, a, b, C])
        k_fft = tf.cast(k_fft, tf.float32)
        k_fft = tf.transpose(k_fft, perm=[0, 3, 1, 2])  
        k_fft = tf.signal.rfft2d(k_fft) 

        attn = q_fft * k_fft
        attn = tf.signal.irfft2d(attn, fft_length=[a, b])
        attn = tf.transpose(attn, perm=[0, 2, 3, 1])
        attn = tf.reshape(attn, [B, N, C])
        attn = self.dropout_attn(attn)
        attn = self.norm(attn)

        out = attn * v 
        out = self.dropout_v(self.to_out(out))

        return out


class FFTFormerEncoder(Layer):
    def __init__(self, dim,  mlp_ratio=4., 
                 h=8, w=5,
                 drop=0.2, ):
        super().__init__()

        self.attn = FFTAttention(dim, drop=drop)
        self.linear = FAFFN(dim=dim, hidden_dim=int(dim * mlp_ratio), h=h, w=w, drop=drop)
        self.drop = Dropout(drop)

    def call(self, x):
        x = x + self.drop(self.attn(x))
        x = x + self.drop(self.linear(x))
        return x


class Block(Layer):
    def __init__(self, dim, mlp_ratio, drop):
        super(Block, self).__init__()
        self.bn = BatchNormalization()
        self.conv = ConvEncoder(dim)
        self.attn= FFTFormerEncoder(dim, mlp_ratio=mlp_ratio,  drop=drop)

    def call(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.attn(x)
        return x


class FFTFormer(Model):
    def __init__(self, dim=16, layer_num=8, mlp_ratio=4,
             num_classes=11, drop=0.2, ):
        super().__init__()

        self.embedding = Conv1D(dim, kernel_size=4, strides=2, padding='same',  use_bias=False)

        self.encoder = [Block(dim=dim, mlp_ratio=mlp_ratio, drop=drop) for _ in range(layer_num)]

        self.avgpool = GlobalAvgPool1D()
        self.head = Dense(num_classes, activation='softmax')


    def call(self, x):

        x = self.embedding(x)

        for encoder in self.encoder:
            x = encoder(x)

        x = self.avgpool(x)
        x = self.head(x)

        return x
    
if __name__ == '__main__':

    model = FFTFormer()
    test_input = tf.random.normal((1, 128, 2))
    output = model(test_input)
    print(output)