from typing import Union
from functools import partial
from keras import layers, Model

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


class HardSigmoid(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.relu6 = layers.ReLU(6.)

    def call(self, inputs, **kwargs):
        x = self.relu6(inputs + 3) * (1. / 6)
        return x


class HardSwish(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.hard_sigmoid = HardSigmoid()

    def call(self, inputs, **kwargs):
        x = self.hard_sigmoid(inputs) * inputs
        return x


def _se_block(inputs, filters, prefix, se_ratio=1 / 4.):
    # [batch, height, width, channel] -> [batch, channel]
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)

    # Target shape. Tuple of integers, does not include the samples dimension (batch size).
    # [batch, channel] -> [batch, 1, 1, channel]
    x = layers.Reshape((1, 1, filters))(x)

    # fc1
    x = layers.Conv2D(filters=_make_divisible(filters * se_ratio),
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)

    # fc2
    x = layers.Conv2D(filters=filters,
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv_1')(x)
    x = HardSigmoid(name=prefix + 'squeeze_excite/HardSigmoid')(x)

    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x,
                        input_c: int,      # input channel
                        kernel_size: int,  # kennel size
                        exp_c: int,        # expanded channel
                        out_c: int,        # out channel
                        use_se: bool,      # whether using SE
                        activation: str,   # RE or HS
                        stride: int,
                        block_id: int,
                        alpha: float = 1.0):

    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)

    input_c = _make_divisible(input_c * alpha)
    exp_c = _make_divisible(exp_c * alpha)
    out_c = _make_divisible(out_c * alpha)

    act = layers.ReLU if activation == "RE" else HardSwish

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
        x = bn(name=prefix + 'expand/BatchNorm')(x)
        x = act(name=prefix + 'expand/' + act.__name__)(x)

    if stride == 2:
        input_size = (x.shape[1], x.shape[2])  # height, width
        x = layers.ZeroPadding2D(padding=correct_pad(input_size, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = bn(name=prefix + 'depthwise/BatchNorm')(x)
    x = act(name=prefix + 'depthwise/' + act.__name__)(x)

    if use_se:
        x = _se_block(x, filters=exp_c, prefix=prefix)

    x = layers.Conv2D(filters=out_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = bn(name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and input_c == out_c:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])

    return x


def DCNN(input_shape=(256, 256, 1),
                       num_classes=12,
                       alpha=1.0,
                       include_top=True):

    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      name="Conv")(img_input)
    x = bn(name="Conv/BatchNorm")(x)
    x = HardSwish(name="Conv/HardSwish")(x)

    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    # input, input_c, k_size, expand_c,output_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, True, "RE", 2, 0)
    x = inverted_cnf(x, 16, 3, 72, 24, False, "RE", 2, 1)
    x = inverted_cnf(x, 24, 3, 88, 24, False, "RE", 1, 2)
    x = inverted_cnf(x, 24, 5, 96, 40, True, "HS", 2, 3)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 4)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 5)
    x = inverted_cnf(x, 40, 5, 120, 48, True, "HS", 1, 6)
    x = inverted_cnf(x, 48, 5, 144, 48, True, "HS", 1, 7)
    x = inverted_cnf(x, 48, 5, 288, 96, True, "HS", 2, 8)

    last_c = _make_divisible(48 * 6 * alpha)
    last_point_c = _make_divisible(1024 * alpha)

    if include_top is True:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, last_c))(x)

        # fc1
        x = layers.Conv2D(filters=last_point_c,
                          kernel_size=1,
                          padding='same',
                          name="Conv_2")(x)
        x = HardSwish(name="Conv_2/HardSwish")(x)

        # fc2
        x = layers.Conv2D(filters=num_classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits/Conv2d_1c_1x1')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x)

    model = Model(img_input, x, name="Dcnn")

    return model