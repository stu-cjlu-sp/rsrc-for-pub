from typing import Union
from functools import partial
from keras import layers, Model, models
import tensorflow as tf

class FourierUnit(layers.Layer):
    def __init__(self, embed_dim, fft_norm='ortho', **kwargs):
        super(FourierUnit, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.fft_norm = fft_norm
        # Adjust the Conv2D layer to ensure it operates correctly on the intended channels
        self.conv_layer = layers.Conv2D(embed_dim * 2, kernel_size=1, strides=1, padding='same')
        self.relu = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        batch = tf.shape(x)[0]
        fft_lengths =[tf.shape(x)[1], tf.shape(x)[2]]

        # Perform FFT with the correct shape handling
        ffted = tf.transpose(x, perm=[0, 3, 1, 2])   # (batch, channels, height, width)
        ffted = tf.signal.rfft2d(ffted, fft_length=None)  # Apply FFT with input height and width
        # Split real and imaginary parts
        ffted_real = tf.math.real(ffted)
        ffted_imag = tf.math.imag(ffted)
        
        # Combine real and imaginary parts along a new dimension
        ffted = tf.stack([ffted_real, ffted_imag], axis=-1)  # (batch, channels, height, width/2+1, 2)
        ffted = tf.transpose(ffted, perm=[0, 1, 4, 2, 3])    # (batch, channels, 2, height, width/2+1)
        
        # Reshape for convolution
        ffted = tf.reshape(ffted, [batch, ffted.shape[1]*ffted.shape[2], ffted.shape[3], ffted.shape[4]])  # (batch, channels*2, height, width/2+1)
        ffted = tf.transpose(ffted, perm=[0, 2, 3, 1])  # (batch, height, width/2+1, channels*2)

        # Apply convolution and activation
        ffted = self.conv_layer(ffted)  # (batch, height, width/2+1, channels*2)
        ffted = self.relu(ffted)
        
        # Reshape back to the original format
        ffted = tf.transpose(ffted, perm=[0, 3, 1, 2])  # (batch, channels*2, height, width/2+1)
        ffted = tf.reshape(ffted, [batch, ffted.shape[1] //2, 2, ffted.shape[2], ffted.shape[3]])  # (batch, channels, 2, height, width/2+1)
        ffted = tf.transpose(ffted, perm=[0, 1, 3, 4, 2])  # (batch, channels, height, width/2+1, 2)
        
        # Convert back to complex form and apply IFFT
        ffted = tf.complex(ffted[..., 0], ffted[..., 1])  # (batch, channels, height, width/2+1)
        ffted = tf.signal.irfft2d(ffted, fft_length=fft_lengths)  # IFFT to get back to (batch, channels, height, width)
        
        output = tf.transpose(ffted, perm=[0, 2, 3, 1])  # (batch, height, width, channels)
        
        return output


# Spectral Transform in Keras
class SpectralTransform(layers.Layer):
    def __init__(self, exp_c,out_c, strides,last_conv=False, **kwargs):
        super(SpectralTransform, self).__init__(**kwargs)
        self.exp_c = exp_c
        self.out_c = out_c
        self.strides = strides
        self.last_conv_flag = last_conv
        
        self.conv1 = models.Sequential([
            layers.Conv2D(exp_c, kernel_size=1, strides=strides, padding='same'),
            layers.LeakyReLU(alpha=0.2)
        ])
        
        self.fu = FourierUnit(exp_c)
        self.conv2 = layers.Conv2D(out_c, kernel_size=1, strides=1, padding='same')
        
        if self.last_conv_flag:
            self.last_conv = layers.Conv2D(out_c, kernel_size=3, strides=1, padding='same')
        else:
            self.last_conv = None

    def call(self, x):
        x_conv1 = self.conv1(x)
        x_f = self.fu(x_conv1)
        output = self.conv2(x_conv1 + x_f)
        if self.last_conv:
            output = self.last_conv(output)
        return output
    
def _make_divisible(ch, divisor=8, min_ch=None):

    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def correct_pad(input_size: Union[int, tuple], kernel_size: int):

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    kernel_size = (kernel_size, kernel_size)

    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(x,
                        input_c: int,      # input channel
                        kernel_size: int,  # kennel size
                        exp_c: int,        # expanded channel
                        out_c: int,        # out channel
                        stride: int,
                        block_id: int,
                        alpha: float = 1.0):

    input_c = _make_divisible(input_c * alpha)
    exp_c = _make_divisible(exp_c * alpha)
    out_c = _make_divisible(out_c * alpha)

    input = x
    shortcut = x
    prefix = 'expanded_conv/'
    if block_id:
        # expand channel
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(filters=exp_c,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    if stride == 2:
        input_size = (x.shape[1], x.shape[2])  # height, width
        x = layers.ZeroPadding2D(padding=correct_pad(input_size, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(filters=out_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)

    if stride == 1 and input_c == out_c:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])

    s = SpectralTransform(exp_c, out_c, strides=stride)(input)
    # s = layers.Conv2D(out_c, kernel_size=1,strides=1,padding='same')(s)

    out = tf.concat([x,s],axis=-1)
    out = layers.Conv2D(out_c, kernel_size=1,strides=1, padding='same')(out)
    return out

def DCNN_SFB(input_shape=(256, 256, 1),
                       num_classes=1000,
                       alpha=1.0,
                       include_top=True):

    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      name="Conv")(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)

    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    # input, input_c, k_size, expand_c,output_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, 2, 0)
    x = inverted_cnf(x, 16, 3, 72, 24, 2, 1)
    x = inverted_cnf(x, 24, 3, 88, 24, 1, 2)
    x = inverted_cnf(x, 24, 5, 96, 40, 2, 3)
    x = inverted_cnf(x, 40, 5, 240, 40, 1, 4)
    x = inverted_cnf(x, 40, 5, 240, 40, 1, 5)
    x = inverted_cnf(x, 40, 5, 120, 48, 1, 6)
    x = inverted_cnf(x, 48, 5, 144, 48, 1, 7)
    x = inverted_cnf(x, 48, 5, 288, 96, 2, 8)
    # x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 9)
    # x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 10)

    last_c = _make_divisible(48 * 6 * alpha)
    last_point_c = _make_divisible(1024 * alpha)

    # x = layers.Conv2D(filters=last_c,
    #                   kernel_size=1,
    #                   padding='same',
    #                   use_bias=False,
    #                   name="Conv_1")(x)
    # x = bn(name="Conv_1/BatchNorm")(x)
    # x = HardSwish(name="Conv_1/HardSwish")(x)

    if include_top is True:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, last_c))(x)

        # fc1
        x = layers.Conv2D(filters=last_point_c,
                          kernel_size=1,
                          padding='same',
                          name="Conv_2")(x)
        x = layers.LeakyReLU(alpha=0.2,name="Conv_2/LeakyReLU")(x)

        # fc2
        x = layers.Conv2D(filters=num_classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits/Conv2d_1c_1x1')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x)

    model = Model(img_input, x, name="MobilenetV3large")

    return model