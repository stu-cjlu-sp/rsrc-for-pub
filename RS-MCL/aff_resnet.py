import tensorflow as tf
from keras import layers, models
from fusion import AFF, iAFF, DAF

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fuse_type='AFF'):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=planes)
        elif fuse_type == 'iAFF':
            self.fuse_mode = iAFF(channels=planes)
        elif fuse_type == 'DAF':
            self.fuse_mode = DAF()
        else:
            self.fuse_mode = None

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.fuse_mode(out, identity)
        out = self.relu(out)

        return out

class ResNet_block(tf.keras.Model):
    def __init__(self, block, resnet_layers, input_shape, num_classes=512, fuse_type='AFF', small_input=False, **kwargs):
        super(ResNet_block, self).__init__(**kwargs)
        self.block_class = block  # 更改名称以避免冲突
        self.resnet_layers = resnet_layers
        self._input_shape = input_shape  # 使用下划线前缀表示私有属性
        self.num_classes = num_classes
        self.fuse_type = fuse_type
        self.small_input = small_input
        self.inplanes = 64

        self.build_model()
        
    def build_model(self):
        inputs = tf.keras.Input(shape=self._input_shape)

        if self.small_input:
            x = layers.Conv2D(self.inplanes, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
        else:
            x = layers.Conv2D(self.inplanes, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        x = self._make_layer(x,self.block_class, 64, self.resnet_layers[0], self.inplanes, fuse_type=self.fuse_type)
        self.inplanes = 64 * self.block_class.expansion  # basic_block 的 expansion 是 1
        x = self._make_layer(x,self.block_class, 128, self.resnet_layers[1], self.inplanes, stride=2, fuse_type=self.fuse_type)
        self.inplanes = 128 * self.block_class.expansion  # basic_block 的 expansion 是 1
        x = self._make_layer(x,self.block_class, 256, self.resnet_layers[2], self.inplanes, stride=2, fuse_type=self.fuse_type)
        self.inplanes = 256 * self.block_class.expansion  # basic_block 的 expansion 是 1
        x = self._make_layer(x,self.block_class, 512, self.resnet_layers[3], self.inplanes, stride=2, fuse_type=self.fuse_type)
        self.inplanes = 512 * self.block_class.expansion  # basic_block 的 expansion 是 1

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.num_classes)(x)
        self.model = tf.keras.Model(inputs, outputs)
        
    def _make_layer(self, x, block, planes, blocks, inplanes, fuse_type='AFF', stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                layers.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization(),
            ])

        layers_list = []
        layers_list.append(block(inplanes, planes, stride, downsample, fuse_type=fuse_type))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(inplanes, planes, fuse_type=fuse_type))

        for layer in layers_list:
            x = layer(x)

        return x

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        # 更新配置字典以包含类的属性
        config = {
            'block': self.block_class.__name__,  # 保存 block 类的名称
            'resnet_layers': self.resnet_layers,
            '_input_shape': self._input_shape,  # 使用下划线前缀表示私有属性
            'num_classes': self.num_classes,
            'fuse_type': self.fuse_type,
            'small_input': self.small_input,
        }

        # 获取父类的配置信息并将其与自定义配置合并
        base_config = super(ResNet_block, self).get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # 确保所有必需的参数都存在
        config.setdefault('_input_shape', (None, None, 1))  # 设置默认值
        block_name = config.pop('block')
        block = globals()[block_name]
        return cls(block=block, input_shape=config.pop('_input_shape'), **config)

def ResNet_aff(im_height, im_width, fuse_type='AFF', **kwargs):
    input_shape = (im_height, im_width, 1)
    inputs = tf.keras.Input(shape=input_shape)
    model = ResNet_block(BasicBlock, resnet_layers=[2, 2, 2, 2], input_shape=input_shape, fuse_type=fuse_type, **kwargs)
    output = model(inputs)
    return tf.keras.Model(inputs, output)
    