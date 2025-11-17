from keras import *
import tensorflow as tf
from keras.models import *
from keras.layers import *
from sklearn.metrics import *


class FeedForward(Layer):
    def __init__(self, dim, hidden_dim, dropout=0., **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(hidden_dim)
        self.swish = Activation("swish")
        self.norm = LayerNormalization(epsilon=1e-5)
        self.dropout = Dropout(dropout)
        self.fc3 = Dense(dim)

    def call(self, x, training=None):

        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1 * self.swish(x2)

        f = self.norm(f1)
        f = self.dropout(f, training=training)
        f= self.fc3(f)

        return f

class MHSA(Layer):
    def __init__(self,
                 d_model=64,
                 d_fix_qk=16,
                 d_fix_v=16,
                 n_head_qk=4,
                 n_head_v=4,
                 dropout=0.,
                 bias=False,
                 talking=True,
                 attn_res=False):
        super(MHSA, self).__init__()
        self.dm = d_model
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h_qk = n_head_qk
        self.h_v = n_head_v
        self.attn_res = attn_res
        self.scale = d_fix_qk ** 0.5

        self.to_q = Dense(d_fix_qk * n_head_qk, use_bias=bias)
        self.to_k = Dense(d_fix_qk * n_head_qk, use_bias=bias)
        self.to_v = Dense(d_fix_v * n_head_v, use_bias=bias)

        self.talking_head1 = Conv2D(filters=n_head_qk, kernel_size=(1, 1), use_bias=False) if talking else Lambda(lambda x: x)
        self.proj_v = Dense(d_model, use_bias=bias)
        self.talking_head2 = Conv2D(filters=n_head_qk, kernel_size=(1, 1), use_bias=False) if talking else Lambda(lambda x: x)

        self.softmax = Softmax(axis=-1)
        self.dropout_attn = Dropout(dropout)
        self.dropout_v = Dropout(dropout)

    def call(self, x, attn_=None, return_attn=False, training=False):
        B = tf.shape(x)[0]
        P = tf.shape(x)[1]

        attn_ = attn_ if attn_ is not None else 0.

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
        attn = tf.transpose(attn, perm=[0, 2, 3, 1])
        attn = self.talking_head1(attn)
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.softmax(attn + attn_)
        attn = tf.transpose(attn, perm=[0, 2, 3, 1])   
        attn = self.talking_head2(attn)                            
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.dropout_attn(attn, training=training)

        x_out = tf.matmul(attn, v)   
        x_out = tf.transpose(x_out, perm=[0, 2, 1, 3])
        x_out = tf.reshape(x_out, (B, P, self.df_v * self.h_v))
        x_out = self.dropout_v(self.proj_v(x_out), training=training)

        if return_attn or self.attn_res:
            return x_out, attn
        return x_out


class Transformer(Layer):
    def __init__(self,
                d_model=64,
                d_fix_qk=64,
                d_fix_v=16,
                d_mid=256,
                n_head_qk=4,
                n_head_v=4,
                dropout=0.,
                bias=False,
                talking=True,
                real_former=True,):
        super(Transformer, self).__init__()
        self.attn_res = real_former

        self.mhsa = MHSA(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
            talking=talking,
            attn_res=real_former
        )

        self.ffn = FeedForward(dim=d_model, hidden_dim=d_mid, dropout=dropout,)

        self.norm1 = LayerNormalization(epsilon=1e-5)
        self.norm2 = LayerNormalization(epsilon=1e-5)

    def call(self, x, attn_=None, training=False):
        if self.attn_res:
            x_sa, attn_ = self.mhsa(x, attn_=attn_, return_attn=True)
        else:
            x_sa = self.mhsa(x, return_attn=False)

        x = self.norm1(x + x_sa)
        x_ffn = self.ffn(x, training=training)
        x = self.norm2(x + x_ffn)
        return x, attn_


class GroupBlock(Model):
    def __init__(self, in_channel, hidden_channel):
        super(GroupBlock, self).__init__()
        self.conv = Sequential([
            Conv1D(hidden_channel, kernel_size=1),
            ReLU(),
            Conv1D(hidden_channel, kernel_size=3, padding='same', groups=in_channel),
            ReLU(),
            Conv1D(in_channel, kernel_size=1)
        ])
        self.shortcut = Sequential([
            Lambda(lambda x: x)
        ])
        self.relu = ReLU()

    def call(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x3 = self.relu(x1 + x2)
        return x3

class TransGroupNet(Model):
    def __init__(self, 
                dim=96,  
                d_fix_qk=12,  
                d_fix_v=12,  
                hidden_features=96 * 2,  
                n_head_qk=8,  
                n_head_v=8, 
                dropout=0.2,  
                layer_num=6,  
                num_class=11,  
                bias=False, 
                talking=True,  
                real_former=False,  
            ):
        super(TransGroupNet,self).__init__()

        self.preconv1 = Sequential([
            Conv1D(dim//3, kernel_size=4, strides=2, name="preconv1"),
            ReLU(),
        ])

        self.preconv2 = Sequential([
            Conv1D(dim//3, kernel_size=4, strides=2,  name="preconv2"),
            ReLU(),
        ])

        self.preconv3 = Sequential([
            Conv1D(dim//3, kernel_size=4, strides=2, name="preconv3"),
            ReLU(),
        ])

        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=[1, 64, dim],
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32
        )
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=[1, 1, dim],
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32
        )

        self.We = self.add_weight(
            name="We",
            shape=[1, dim, dim],
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True,
            dtype=tf.float32
        )

        self.convgroup = Sequential([
            GroupBlock(64, 64),
            GroupBlock(64, 64)
        ])

        self.dropout = Dropout(0.1)
        self.encoders = [Transformer(
            d_model=dim,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            d_mid=hidden_features,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
            talking=talking,
            real_former=real_former,
        ) for _ in range(layer_num)]

        self.fc = Dense(num_class, activation='softmax')


    def call(self, input):

        x1 = input[:, :, 0:1]
        x2 =input[:, :, 1:2]
        x3 =input[:, :, 2:3]

        A = self.preconv1(x1)  
        P = self.preconv2(x2)
        F = self.preconv3(x3)
        x = Concatenate(axis=-1)([A, P, F])

        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        cls_tokens = tf.tile(self.cls_token, [B, 1, 1])
        x = tf.matmul(x, self.We)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embed[:, :(N + 1)]
        x = self.dropout(x)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.convgroup(x)
        x = tf.transpose(x, perm=[0, 2, 1])

        attn_res = None
        for encoder in self.encoders:
            x, attn_res = encoder(x, attn_res)

        return self.fc(x[:, 0])


if __name__ == "__main__":
    x = tf.random.normal((1, 128, 3))
    model = TransGroupNet()
    output = model(x)
    print(output.shape)