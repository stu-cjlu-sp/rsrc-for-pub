import tensorflow as tf
from keras import layers, models
from functools import partial
from typing import Union

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
    def __init__(self, embed_dim, strides,last_conv=False, **kwargs):
        super(SpectralTransform, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.strides = strides
        self.last_conv_flag = last_conv
        
        self.conv1 = models.Sequential([
            layers.Conv2D(embed_dim//2, kernel_size=1, strides=strides, padding='same'),
            layers.LeakyReLU(alpha=0.2)
        ])
        
        self.fu = FourierUnit(embed_dim//2)
        self.conv2 = layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding='same')
        
        if self.last_conv_flag:
            self.last_conv = layers.Conv2D(embed_dim, kernel_size=3, strides=1, padding='same')
        else:
            self.last_conv = None

    def call(self, x):
        x_conv1 = self.conv1(x)
        x_f = self.fu(x_conv1)
        output = self.conv2(x_conv1 + x_f)
        if self.last_conv:
            output = self.last_conv(output)
        return output

# Residual Block in Keras
class ResB(layers.Layer):
    def __init__(self, embed_dim, ratio=1, stride=1, **kwargs):
        super(ResB, self).__init__(**kwargs)
        self.body = models.Sequential([
            layers.Conv2D(embed_dim * ratio, kernel_size=1, strides=1, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding='same')
        ])

    def call(self, x):
        return x + self.body(x)

# SFB (Spectral Fusion Block) in Keras
class SFB(layers.Layer):
    def __init__(self, embed_dim, ratio=1, **kwargs):
        super(SFB, self).__init__(**kwargs)
        self.S = ResB(embed_dim, ratio)
        self.F = SpectralTransform(embed_dim)
        self.fusion = layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding='same')

    def call(self, x):
        s = self.S(x)
        f = self.F(x)
        out = tf.concat([s, f], axis=-1)
        out = self.fusion(out)
        return out

  
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


def inverted_res_block(x,
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

    s = SpectralTransform(out_c, strides=stride)(input)
    # s = layers.Conv2D(out_c, kernel_size=1,strides=1,padding='same')(s)

    out = tf.concat([x,s],axis=-1)
    out = layers.Conv2D(out_c, kernel_size=1,strides=1,padding='same')(out)
    return out
