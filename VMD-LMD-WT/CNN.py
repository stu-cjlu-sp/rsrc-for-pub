import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D, Activation, \
     Dropout, BatchNormalization, Concatenate

def CNN(im_height=112, im_width=112):
    input_image = Input(shape=(im_height, im_width, 1), dtype="float64")
    x = Conv2D(64, (7, 7), padding='same')(input_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    def inception_v1(y):
        y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)
        y2 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y2 = Activation('relu')(y2)
        y3 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y3 = Activation('relu')(y3)
        y2 = Conv2D(16, kernel_size=(2, 2), strides=(2, 2), padding='same')(y2)
        y2 = Activation('relu')(y2)
        y3 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(y3)
        y3 = Activation('relu')(y3)
        y3 = Conv2D(16, kernel_size=(2, 2), strides=(2, 2), padding='same')(y3)
        y3 = Activation('relu')(y3)
        y = Concatenate(axis=3)([y1, y2, y3])
        return y  

    def module(y):
        y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)
        y1 = Conv2D(64, (3, 3), padding='same',dilation_rate=1)(y)
        y1 = BatchNormalization()(y1)
        y1 = Activation('relu')(y1)
        y_conv = Conv2D(64, (1, 1), padding='same')(y)
        y_conv = BatchNormalization()(y_conv)
        y_conv = Activation('relu')(y_conv)
        y3 = Concatenate()([y1, y_conv])
        y4 = Conv2D(64, (1, 1), padding='same')(y3)
        y4 = BatchNormalization()(y4)
        y4 = Activation('relu')(y4)
        y5 = Conv2D(64, (3, 3), padding='same',dilation_rate=2)(y4)
        y5 = BatchNormalization()(y5)
        y5 = Activation('relu')(y5)
        y_conv2 = Conv2D(64, (1, 1), padding='same')(y4)
        y_conv2 = BatchNormalization()(y_conv2)
        y_conv2 = Activation('relu')(y_conv2)
        y7 = Concatenate()([y5, y3, y_conv2])
        y8 = Conv2D(64, (1, 1), padding='same')(y7)
        y8 = BatchNormalization()(y8)
        y8 = Activation('relu')(y8)
        y9 = Conv2D(64, (3, 3), padding='same',dilation_rate=4)(y8)
        y9 = BatchNormalization()(y9)
        y9= Activation('relu')(y9)
        y_conv3 = Conv2D(64, (1, 1), padding='same')(y8)
        y_conv3 = BatchNormalization()(y_conv3)
        y_conv3 = Activation('relu')(y_conv3)
        
        y_Concatenate3 = Concatenate()([y9, y7, y_conv3])
        y11 = Conv2D(64, (1, 1), padding='same')(y_Concatenate3)
        y11 = BatchNormalization()(y11)
        y11 = Activation('relu')(y11)
        y = Concatenate()([y, y11])
        return y
    
    x = inception_v1(x)
    x = module(x)
    x = module(x)
    x = module(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = Dense(12)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=input_image, outputs=x_out, name='cnn')
    return model
