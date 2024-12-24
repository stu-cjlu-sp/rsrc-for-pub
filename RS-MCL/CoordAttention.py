import tensorflow as tf

def CoordAtt(x, reduction=32):
    def coord_act(x):
        tmpx = tf.nn.relu6(x + 3) / 6
        x = x * tmpx
        return x

    x_shape = tf.TensorShape(x.shape).as_list()
    b, h, w, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

    x_h = tf.keras.layers.AveragePooling2D(pool_size=(1, w), strides=1)(x)
    x_w = tf.keras.layers.AveragePooling2D(pool_size=(h, 1), strides=1)(x)
    x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)

    y = tf.keras.layers.Conv2D(mip, (1, 1), strides=1, padding='valid', activation=coord_act)(y)

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])

    a_h = tf.keras.layers.Conv2D(c, (1, 1), strides=1, padding='valid', activation=tf.nn.sigmoid)(x_h)
    a_w = tf.keras.layers.Conv2D(c, (1, 1), strides=1, padding='valid', activation=tf.nn.sigmoid)(x_w)

    out = x * a_h * a_w
    return out