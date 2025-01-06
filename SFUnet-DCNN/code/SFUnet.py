import tensorflow as tf
from keras.optimizers import Adam
from tensorflow import *
from keras.models import *
from keras.layers import *
from layer_utils import *
from keras import Model
from Transformer_layers import *
from SFB import *
from load_dataset import *
from cosine_annealing import CosineAnnealingScheduler
from keras_flops import get_flops
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# import tensorflow as tf
# from tensorflow.python.profiler.model_analyzer import profile
# from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# def get_flops(model):
#     forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
#     graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
#     flops = graph_info.total_float_ops
#     return flops

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp,block_id, shift_window=True, name='' ):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0 # Droupout after each MLP layer
    attn_drop_rate = 0 # Dropout after Swin-Attention
    proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    inputs = X

    for i in range(stack_num):

        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = SwinTransformerBlock(dim=embed_dim, 
                                             num_patch=num_patch, 
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             shift_size=shift_size_temp, 
                                             num_mlp=num_mlp, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, 
                                             attn_drop=attn_drop_rate, 
                                             proj_drop=proj_drop_rate, 
                                             drop_path_prob=drop_path_rate, 
                                             name='name{}'.format(i))(X)
    X = patch_to_image(num_patch=num_patch)(X)
    X = inverted_res_block(X, input_c=embed_dim, kernel_size=3, exp_c=embed_dim, out_c=embed_dim, stride=1,block_id=block_id)
    X = image_to_patch(num_patch=num_patch )(X)
    outputs = Add()([X, inputs])    
    return outputs

def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):

    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    id =0

    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim, 
                               num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               block_id=id,
                               name='{}_swin_down0'.format(name))
    id+=1
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_down, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], 
                                   window_size=window_size[i+1], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   block_id=id,
                                   name='{}_swin_down{}'.format(name, i+1))
        id+=1

        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, 
                                               return_vector=True)(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_up, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i], 
                                   window_size=window_size[i], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   block_id=id,
                                   name='{}_swin_up{}'.format(name, i))
        id+=1
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim, 
                                           upsample_rate=patch_size[0], 
                                           return_vector=False)(X)
    
    return X

def get_model():

    filter_num_begin = 64 # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4             # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    stack_num_down = 2         # number of Swin Transformers per downsampling level
    stack_num_up = 2           # number of Swin Transformers per upsampling level
    patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    num_heads = [4, 8, 16, 32]   # number of attention heads per down/upsampling level
    window_size = [8, 8, 8, 8] # the size of attention window per down/upsampling level
    num_mlp = 512             # number of MLP nodes within the Transformer
    shift_window= True          # Apply window shifting, i.e., Swin-MSA

    input_size = (256,256,1)
    
    input= Input(input_size)

    x = swin_unet_2d_base(input, filter_num_begin, depth, stack_num_down, stack_num_up, 
                          patch_size, num_heads, window_size, num_mlp, 
                          shift_window=shift_window, name='swin_unet')
    
    x = Conv2D(filters=1, kernel_size=3, padding='same')(x)

    output = Activation('sigmoid')(x)    

    model = Model(inputs=[input,], outputs=[output,])

    return model

def train_model():

    starttime = datetime.datetime.now()

    model = get_model()

    model.build((None, 256, 256,1))

    model.summary()

    flops = get_flops(model)
    print(f"FLOPS: {flops / 10 ** 9:.05} G")

    data, data_noise = input_data_train_Unet()

    validation_data, validation_data_noise = input_data_validation_Unet()

    optimizer = Adam(0.0001)

    model.compile(
                        loss=['mae'],
                        optimizer = optimizer,
                        metrics = ['mse','mae']
                        )
    
    reduce_lr = CosineAnnealingScheduler(T_max=200, eta_max=1e-4, eta_min=1e-6)
    
    model.fit(x=data_noise, y=data, validation_data=(validation_data_noise , validation_data), batch_size=8, epochs=1, verbose=1, callbacks=[reduce_lr])

    model.save('/home/sp432cy/sp432cy/Final_Model/SFUnet', save_format='tf')

    endtime = datetime.datetime.now()
    print('Model training time:', (endtime - starttime).seconds, 's')

if __name__ == '__main__':
    
    train_model()


