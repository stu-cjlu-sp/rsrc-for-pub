import os
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import math
from torch.autograd import Variable

def stemIQ(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, D]
    """
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs//2 , kernel_size=5, stride=1, padding=2,groups=in_chs),
        nn.BatchNorm1d(out_chs//2),
        )
 
class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D/stride]
    """

    def __init__(self, patch_size=3, stride=1, padding=1,
                 in_chans=2, embed_dim=768, norm_layer=nn.BatchNorm1d):
        super().__init__()
        patch_size = patch_size
        stride = stride
        padding = padding
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block for 1D inputs.
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()              # x: [B, C, D]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class FEE(nn.Module):
    def __init__(self, input_channel, drop):
        super(FEE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, input_channel, kernel_size=1),  # pointwise conv
            nn.Conv1d(input_channel, input_channel, kernel_size=5, padding=2, groups=input_channel),  # depthwise conv
            nn.BatchNorm1d(input_channel),
            nn.GELU(),
            nn.Conv1d(input_channel, input_channel, kernel_size=1),  # another pointwise conv
            nn.BatchNorm1d(input_channel),
            nn.GELU(),
            nn.Conv1d(input_channel, input_channel, kernel_size=3, padding=1, groups=1),         
)

        self.se = SELayer(input_channel)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.drop(out)
        return out + x  
    

class ConvEncoder(nn.Module):
    """
    Input: tensor with shape [B, C, D]
    Output: tensor with shape [B, C, D]
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
    
class FCN(nn.Module):
    """
    Input: tensor with shape [B, C, D]
    Output: tensor with shape [B, C, D]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class RoPE(nn.Module):
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution, dim))

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

        return x
    
class MLLABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        H = self.input_resolution
        assert L == H, "input feature has wrong size"

        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)

        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class LocalRepresentation(nn.Module):
    """
   
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """

    def __init__(self, dim, kernel_size=5, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
    


class MILAFormer_Encoder(nn.Module):
    """
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = LocalRepresentation(dim=dim, kernel_size=5, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = MLLABlock(dim=dim, input_resolution=128)
        self.linear = FCN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 1)).permute(0, 2, 1))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 1)).permute(0, 2, 1))
            x = x + self.drop_path(self.linear(x))
        return x


def Stage(dim, index, layers, mlp_ratio=4.,
          act_layer=nn.GELU,
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    """
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        if layers[index] - block_idx <= vit_num:
            blocks.append(MILAFormer_Encoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))

        else:
            blocks.append(ConvEncoder(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))
    blocks = nn.Sequential(*blocks)
    return blocks


class MILAFormer(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4,
                 act_layer=nn.GELU,
                 num_classes=11,
                 down_patch_size=5, down_stride=3, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 vit_num=1,
                 ):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.BN = nn.BatchNorm1d(2)
        self.patch_embedIQ = stemIQ(2, embed_dims[0]//4)
        self.fee = FEE(embed_dims[0]//8,drop_rate)
        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          act_layer=act_layer,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)
        self.patch_LSTM = nn.LSTM(input_size=embed_dims[0]//8, hidden_size=embed_dims[0]//2,bidirectional=True, batch_first=True, num_layers=2, dropout=drop_rate)

        # Classifier head
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()
        self.apply(self._init_weights)
        self.globalmaxpool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.globalavgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x
    
    
    def forward(self, x):
        x = self.BN(x)
        x = self.patch_embedIQ(x)
        x = self.fee(x)  
        x,_ = self.patch_LSTM(x.permute(0,2,1))
        x = self.forward_tokens(x.permute(0,2,1))
        x = self.norm(x)
        cls_out = self.head(self.globalavgpool(x))
        return cls_out