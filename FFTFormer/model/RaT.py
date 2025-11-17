from keras import *
from keras.models import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf

class MLP(Layer):
    
    def __init__(self, d_model, dim_feedforward, dropout=0., **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.fc1 = Dense(dim_feedforward)
        self.gelu = Activation("gelu")
        self.dropout = Dropout(dropout)
        self.fc2 = Dense(d_model)

    def call(self, x, training=None):

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)

        return x


class GlobalFilter(Layer):
    def __init__(self, dim, h=4, w=3):
        super(GlobalFilter, self).__init__()
        self.h = h
        self.w = w
        self.dim = dim

        self.complex_weight  = self.add_weight(
            shape=(h, w, dim, 2),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name='real_weight'
        )

    def call(self, x, spatial_size=None):

        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = tf.shape(x)[2]

        if spatial_size is None:
            a = b = tf.cast(tf.math.sqrt(tf.cast(N, tf.float32)), tf.int32)
        else:
            a, b = spatial_size

        x = tf.reshape(x, [B, a, b, C])
        x = tf.cast(x, tf.float32)
        x = tf.transpose(x, perm=[0, 3, 1, 2])  
        x = tf.signal.rfft2d(x, fft_length=None) 
        x = tf.transpose(x, perm=[0, 2, 3, 1])  

        complex_weight = tf.complex(self.complex_weight[..., 0], self.complex_weight[..., 1])
        x = x * complex_weight

        x = tf.transpose(x, perm=[0, 3, 1, 2]) 
        x = tf.signal.irfft2d(x, fft_length=[a, b])
        x = tf.transpose(x, perm=[0, 2, 3, 1])  

        x = tf.reshape(x, [B, N, C])

        return x


class MHSA(Layer):
    def __init__(self,
                 d_model=64,
                 d_fix_qk=16,
                 d_fix_v=16,
                 n_head_qk=4,
                 n_head_v=4,
                 dropout=0.,
                 bias=False,
                ):
        super(MHSA, self).__init__()
        self.dm = d_model
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h_qk = n_head_qk
        self.h_v = n_head_v
        self.scale = d_fix_qk ** 0.5

        self.to_q = Dense(d_fix_qk * n_head_qk, use_bias=bias)
        self.to_k = Dense(d_fix_qk * n_head_qk, use_bias=bias)
        self.to_v = Dense(d_fix_v * n_head_v, use_bias=bias)

        self.proj_v = Dense(d_model, use_bias=bias)

        self.softmax = Softmax(axis=-1)
        self.dropout_attn = Dropout(dropout)
        self.dropout_v = Dropout(dropout)

    def call(self, x, training=False):

        B = tf.shape(x)[0]
        P = tf.shape(x)[1]

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = tf.reshape(q, (B, P, self.h_qk, self.df_qk))
        k = tf.reshape(k, (B, P, self.h_qk, self.df_qk))
        v = tf.reshape(v, (B, P, self.h_v, self.df_v))

        q = tf.transpose(q, perm=[0, 2, 1, 3]) / self.scale  
        k = tf.transpose(k, perm=[0, 2, 1, 3])              
        v = tf.transpose(v, perm=[0, 2, 1, 3])                

        attn = tf.matmul(q, k, transpose_b=True)             
        attn = self.softmax(attn)
        attn = self.dropout_attn(attn, training=training)

        x_out = tf.matmul(attn, v)                           
        x_out = tf.transpose(x_out, perm=[0, 2, 1, 3])       
        x_out = tf.reshape(x_out, (B, P, self.df_v * self.h_v))

        x_out = self.dropout_v(self.proj_v(x_out), training=training)

        return x_out


class FrequencyEncoder(Layer):
    def __init__(self,
                 d_model=64,
                 d_mid=256,
                 h=3,
                 w=2,
                 dropout=0.,
                ):
        super(FrequencyEncoder, self).__init__()

        self.mhsa = GlobalFilter(d_model, h=h, w=w)
   
        self.ffn = MLP(d_model=d_model, dim_feedforward=d_mid, dropout=dropout,)

        self.norm1 = LayerNormalization(epsilon=1e-5)
        self.norm2 = LayerNormalization(epsilon=1e-5)

    def call(self, x,  training=False):

        x_sa = self.mhsa(x)
        x = self.norm1(x + x_sa)

        x_ffn = self.ffn(x, training=training)
        x = self.norm2(x + x_ffn)

        return x


class AttentionEncoder(Layer):
    def __init__(self,
                 d_model=64,
                 d_fix_qk=64,
                 d_fix_v=16,
                 d_mid=256,
                 n_head_qk=4,
                 n_head_v=4,
                 dropout=0.,
                 bias=False,
                 ):
        super(AttentionEncoder, self).__init__()

        self.mhsa = MHSA(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
        )

        self.ffn = MLP(d_model=d_model, dim_feedforward=d_mid, dropout=dropout,)

        self.norm1 = LayerNormalization(epsilon=1e-5)
        self.norm2 = LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):

        x_sa = self.mhsa(x)
        x = self.norm1(x + x_sa)
        x_ffn = self.ffn(x, training=training)
        x = self.norm2(x + x_ffn)

        return x


class Frame(Layer):  
    def __init__(self, PatchSize=32, n_patch=32, overlap=0.5, bias=False):
        super(Frame, self).__init__()
        self.PatchSize = PatchSize
        self.Stride = int(PatchSize * overlap)
        self.FrameNum = n_patch - 1
        self.embedding = Dense(PatchSize * 2, use_bias=bias)

    def call(self, x):
        input_feature = []
        for i in range(self.FrameNum):
            start = i * self.Stride
            end = start + self.PatchSize
            patch_I = x[:, start:end, 0]  
            patch_Q = x[:, start:end, 1]
            patch = tf.concat([patch_I, patch_Q], axis=-1)
            input_feature.append(patch)

        input_feature = tf.stack(input_feature, axis=1)

        return self.embedding(input_feature)


class RaT(Model):
    def __init__(self,
                 patch_size=16,
                 d_fix_qk=16,
                 d_fix_v=16,
                 seq_length=128,
                 hidden_features=64 * 4,  
                 n_head_qk=4,
                 n_head_v=4,
                 overlap=0.5, 
                 dropout=0.2,  
                 layer_num1=2,  
                 layer_num2=10,
                 num_class=11, 
                 bias=False,  
                 ):
        super(RaT, self).__init__()

        n_patch = int((seq_length - patch_size) / int(patch_size * overlap) + 2)
        in_features = patch_size *2

        self.Embedding = Frame(patch_size, n_patch, bias=False, overlap=overlap)

        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=[1, n_patch, in_features],
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=[1, 1, patch_size * 2],
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        self.dropout = Dropout(0.2)
        self.encoder1 = [FrequencyEncoder(
            d_model=in_features,
            d_mid=hidden_features,
            h=4,
            w=3,
            dropout=dropout,
        ) for _ in range(layer_num1)]

        self.encoder2 = [AttentionEncoder(
            d_model=in_features,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            d_mid=hidden_features,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
        ) for _ in range(layer_num2)]

        self.classifier = Sequential([
            GlobalAvgPool1D(),
            LayerNormalization(epsilon=1e-5),
            Dropout(0.1),
            Dense(num_class, activation='softmax')
        ])


    def call(self, x):
        B = tf.shape(x)[0]
        x = self.Embedding(x)
        cls_token = tf.tile(self.cls_token, [B, 1, 1])
        x = tf.concat((x, cls_token),  axis=1)
        x += self.pos_embed
        x = self.dropout(x)

        for encoder1 in self.encoder1:
            x = encoder1(x)

        for encoder2 in self.encoder2:
            x = encoder2(x)

        return self.classifier(x)

if __name__ == '__main__':

    model = RaT()
    test_input = tf.random.normal((1, 128, 2))
    output = model(test_input)
    print(output)