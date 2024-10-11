from keras import layers
import tensorflow as tf
from keras import backend

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)

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
    
def CBAM_Block(input_layer, filter_num, reduction_ratio=32, kernel_size=7,  name=None):
	"""CBAM: Convolutional Block Attention Module Block
    Args:
      input_layer: input tensor
      filter_num: integer, number of neurons in the hidden layers
      reduction_ratio: integer, default 32,reduction ratio for the number of neurons in the hidden layers
      kernel_size: integer, default 7, kernel size of the spatial convolution excitation convolution
      name: string, block label
    Returns:
      Output A tensor for the CBAM attention block
    """
	axis = -1

	# CHANNEL ATTENTION
	avg_pool = layers.GlobalAveragePooling2D(name=name + "_Chanel_AveragePooling")(input_layer)
	max_pool = layers.GlobalMaxPool2D(name=name + "_Chanel_MaxPooling")(input_layer)

	# Shared MLP
	# dense1 = layers.Dense(filter_num // reduction_ratio, activation='relu', name=name + "_Chanel_FC_1")
	# dense2 = layers.Dense(filter_num, name=name + "_Chanel_FC_2")

	# avg_out = dense2(dense1(avg_pool))
	# max_out = dense2(dense1(max_pool))

	avg_pool = layers.Reshape((1, 1, filter_num))(avg_pool)
	max_pool = layers.Reshape((1, 1, filter_num))(max_pool)

    # 共享卷积层
	conv1 = layers.Conv2D(filter_num // reduction_ratio, kernel_size=1, padding='same', activation='relu', name=name + "_Chanel_Conv1")
	conv2 = layers.Conv2D(filter_num, kernel_size=1, padding='same', name=name + "_Chanel_Conv2")

	avg_out = conv2(conv1(avg_pool))
	max_out = conv2(conv1(max_pool))

	channel = layers.add([avg_out, max_out])
	channel = layers.Activation('sigmoid', name=name + "_Chanel_Sigmoid")(channel)
	channel = layers.Reshape((1, 1, filter_num), name=name + "_Chanel_Reshape")(channel)
	channel_output = layers.multiply([input_layer, channel])

	# SPATIAL ATTENTION
	avg_pool2 = tf.reduce_mean(channel_output, axis=axis, keepdims=True)
	max_pool2 = tf.reduce_max(channel_output, axis=axis, keepdims=True)

	spatial = layers.concatenate([avg_pool2, max_pool2], axis=axis)

	# K = 7 achieves the highest accuracy
	spatial = layers.Conv2D(1, kernel_size=kernel_size, padding='same', name=name + "_Spatial_Conv2D")(spatial)
	spatial = layers.Activation('sigmoid', name=name + "_Spatial_Sigmoid")(spatial)

	spatial_out =layers.multiply([channel_output, spatial])

	CBAM_out = layers.add([input_layer, spatial_out])

	return CBAM_out