import tensorflow as tf
from CoordAttention import CoordAtt
def BasicBlock1(inputs, num_channels, kernel_size, num_blocks, skip_blocks, regularizer):
    """Basic residual block
    
    This creates residual blocks of ConvNormRelu for num_blocks.

    Args:
        inputs: 4-D tensor [B, W, H, CH]
        num_channels: int, number of convolutional filters
        kernel_size: int, size of kernel
        num_blocks: int, number of consecutive 
        skip_blocks: int, this block will be skipped. Used for when stride is >1
        regularizer: tensorflow regularizer
        name: name of the layer
    Returns:
        x: 4-D tensor of the image activation [B, W, H, CH]
    """
    
    x = inputs

    for i in range(num_blocks):
        if i not in skip_blocks:
            x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[1,1], regularizer=regularizer)
            x1 = CoordAtt(x1)
            x = tf.keras.layers.concatenate([x, x1])
            x = tf.keras.layers.Activation('relu')(x)
    return x

def BasicBlockDown(inputs, num_channels, kernel_size, regularizer):
    """Single residual block with strided downsampling
    
    Args:
        inputs: 4-D tensor [B, W, H, CH]
        num_channels: int, number of convolutional filters
        kernel_size: int, size of kernel
        regularizer: tensorflow regularizer
        name: name of the layer
    Returns:
        x: 4-D tensor of the image activation [B, W, H, CH]
    """
    
    x = inputs
    x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[2,1], regularizer=regularizer)
    x = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=2, padding='same', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.Activation('relu')(x)
    return x   

def ResNet_ca(im_height, im_width, weight_decay=None):
    
    inputs = tf.keras.layers.Input(shape=(im_height, im_width, 1), dtype="float64")
    if weight_decay:
        regularizer = tf.keras.regularizers.l2(weight_decay) #not valid for Adam, must use  AdamW
    else:
        regularizer = None

    x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(x)
    x = CoordAtt(x)
    x = BasicBlock1(x, num_channels=64, kernel_size=3, num_blocks=2, skip_blocks=[], regularizer=regularizer)
    x = CoordAtt(x)
    x = BasicBlock1(x, num_channels=128, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer)
    x = CoordAtt(x)
    x = BasicBlock1(x, num_channels=256, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer)
    x = CoordAtt(x)
    x = BasicBlock1(x, num_channels=512, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer)
    x = CoordAtt(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=512, use_bias=True, activation='linear', kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inputs, x)
    return model