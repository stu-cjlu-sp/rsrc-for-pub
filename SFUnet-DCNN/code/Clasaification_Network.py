from keras import layers
import tensorflow as tf
from keras.layers import Layer

class RACN(Layer):
    def __init__(self, filter_num, reduction_ratio=32, kernel_size=7, name=None, **kwargs):
        """
        CBAM: Convolutional Block Attention Module Block
        Args:
          filter_num: Integer, number of neurons in the hidden layers.
          reduction_ratio: Integer, default 32, reduction ratio for the number of neurons in the hidden layers.
          kernel_size: Integer, default 7, kernel size of the spatial convolution excitation convolution.
          name: String, block label.
        """
        super(RACN, self).__init__(name=name, **kwargs)
        self.filter_num = filter_num
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        # Define channel attention layers
        self.global_avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_Channel_AvgPooling")
        self.global_max_pool = layers.GlobalMaxPooling2D(name=f"{name}_Channel_MaxPooling")
        self.conv1 = layers.Conv2D(filter_num // reduction_ratio, kernel_size=1, padding='same', activation='relu',
                                   name=f"{name}_Channel_Conv1")
        self.conv2 = layers.Conv2D(filter_num, kernel_size=1, padding='same', name=f"{name}_Channel_Conv2")
        self.sigmoid = layers.Activation('sigmoid', name=f"{name}_Channel_Sigmoid")

        # Define spatial attention layers
        self.spatial_conv = layers.Conv2D(1, kernel_size=kernel_size, padding='same', name=f"{name}_Spatial_Conv2D")
        self.spatial_sigmoid = layers.Activation('sigmoid', name=f"{name}_Spatial_Sigmoid")

        # Fully connected layers
        self.fc_conv1 = layers.Conv2D(filters=288, kernel_size=1, padding='same', use_bias=False, name=f"{name}_FC_Conv1")
        self.fc_conv2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', name=f"{name}_FC1")
        self.fc_conv3 = layers.Conv2D(filters=12, kernel_size=1, padding='same', name=f"{name}_FC2")
        self.flatten = layers.Flatten(name=f"{name}_FC2_Flatten")
        self.softmax = layers.Softmax(name=f"{name}_Predictions")

    def call(self, inputs, **kwargs):
        # Channel Attention
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)

        avg_pool = layers.Reshape((1, 1, self.filter_num))(avg_pool)
        max_pool = layers.Reshape((1, 1, self.filter_num))(max_pool)

        avg_out = self.conv2(self.conv1(avg_pool))
        max_out = self.conv2(self.conv1(max_pool))

        channel_attention = self.sigmoid(layers.add([avg_out, max_out]))
        channel_output = layers.multiply([inputs, channel_attention])

        # Spatial Attention
        avg_pool_spatial = tf.reduce_mean(channel_output, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_output, axis=-1, keepdims=True)

        spatial_attention = layers.concatenate([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.spatial_sigmoid(self.spatial_conv(spatial_attention))

        spatial_output = layers.multiply([channel_output, spatial_attention])

        # Fully connected layers
        cbam_output = layers.add([inputs, spatial_output])
        fc = self.fc_conv1(cbam_output)
        fc = layers.LeakyReLU(alpha=0.2)(fc)

        fc = layers.GlobalAveragePooling2D()(fc)
        fc = layers.Reshape((1, 1, 288))(fc)

        fc1 = self.fc_conv2(fc)
        fc1 = layers.LeakyReLU(alpha=0.2)(fc1)

        fc2 = self.fc_conv3(fc1)
        fc2 = self.flatten(fc2)

        return self.softmax(fc2)
