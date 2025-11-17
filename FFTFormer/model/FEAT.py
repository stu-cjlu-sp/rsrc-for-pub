from keras import *
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.activations import *

class DB_GLU(Layer):
    def __init__(self,
                d_model=64,
                dim_feedforward=128,
                dropout=0.1,
                ):
        super(DB_GLU, self).__init__()
        self.dim_f = dim_feedforward // 2

        self.linear1 = Dense(dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.linear2 = Dense(d_model)

        self.Act = Activation("gelu")

    def call(self, x, training=False):

        x_ = self.linear1(x)
        x_1 = x_[..., :self.dim_f]
        x_2 = x_[..., self.dim_f:]

        out = x_1 * self.Act(x_2) + x_2 * self.Act(x_1)
        return self.linear2(self.dropout1(out, training=training))

class MHSA(Layer):
    def __init__(self,
                d_model=64,
                d_fix_qk=16,
                d_fix_v=16,
                n_head_qk=4,
                n_head_v=4,
                dropout=0.,
                bias=False,
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

        self.proj_bf = Conv2D(filters=n_head_qk, kernel_size=(1, 1), use_bias=False) 
        self.proj_v = Dense(d_model, use_bias=bias)

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
        attn = self.proj_bf(attn)                          
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.softmax(attn + attn_)
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
            attn_res=real_former
        )

        self.ffn = DB_GLU(d_model=d_model, dim_feedforward=d_mid, dropout=dropout,)

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


class FEAT(Model):
    def __init__(self,
                patch_size=16,
                d_fix_qk=16,  
                d_fix_v=16,  
                seq_length=128,  
                hidden_features=64 * 4,  
                n_head_qk=4,  
                n_head_v=4,  
                overlap=0.2,  
                dropout=0., 
                layer_num=8,  
                num_class=11,  
                bias=False,  
                real_former=False, 
                ):
        
        super(FEAT, self).__init__()
        n_patch = int((seq_length - patch_size) / int(patch_size * overlap) + 2)
        in_features = patch_size * 2

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
        self.dropout = Dropout(0.1)

        self.Embedding = Frame(patch_size, n_patch, bias=False, overlap=overlap)
        self.encoders = [Transformer(
            d_model=in_features,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            d_mid=hidden_features,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
            real_former=real_former,
        ) for _ in range(layer_num)]
        self.classifier = Dense(num_class,activation="softmax")

    def call(self, x):
        B = tf.shape(x)[0]
        x = self.Embedding(x)  
        cls_token = tf.tile(self.cls_token, [B, 1, 1])
        x = tf.concat((x, cls_token),  axis=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        attn_res = None
        for encoder in self.encoders:
            x, attn_res = encoder(x, attn_res)
        return self.classifier(x[:, -1])
    
if __name__ == '__main__':
    model = FEAT()
    test_input = tf.random.normal((1, 128, 2))
    output = model(test_input)
    print(output)