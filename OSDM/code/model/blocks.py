import torch
import torch.nn as nn
from mamba_ssm import Mamba


def FNN(model_dim, dff):
    return nn.Sequential(
        nn.Linear(model_dim, dff),
        nn.ReLU(),
        nn.Linear(dff, model_dim)
    )


class MambaBlock(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.mamba = Mamba(
            model_dim,
            d_state=16,
            expand=4,
        )

    def forward(self, x):
        return self.mamba(x)


class Mambalayer(nn.Module):
    def __init__(self, model_dim, ddf, dropout_rate):
        super().__init__()
        self.mamba = MambaBlock(model_dim)
        self.ffn = FNN(model_dim, ddf)
        self.ln1 = nn.LayerNorm(model_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(model_dim, eps=1e-6)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        mamba_out = self.drop1(self.mamba(x))
        out1 = self.ln1(x + mamba_out)
        f_out = self.drop2(self.ffn(out1))
        out2 = self.ln2(out1 + f_out)
        return out2


class OverlappingPatchSplitter(nn.Module):
    def __init__(self, max_len, p_size, overlap):
        super(OverlappingPatchSplitter, self).__init__()
        self.p_size = p_size
        self.overlap = overlap
        self.stride = p_size - overlap
        self.max_len = max_len

    def forward(self, x3):
        x3 = x3.unsqueeze(3)
        x3 = x3.permute(0, 2, 1, 3)
        patches = x3.unfold(2, self.p_size, self.stride)
        patches = patches.squeeze(3)
        patches = patches.permute(0, 2, 3, 1)
        b, n_patch, p_size, d = patches.shape
        out = patches.reshape(b * n_patch, p_size, d)
        
        return out


class SparseProjection(nn.Module):
    def __init__(self, in_features, out_features, sparsity):
        super(SparseProjection, self).__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.drop = nn.Dropout(p=sparsity)

        eye = torch.eye(out_features)[:in_features, :]
        self.proj.weight.data = eye.t()

        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, x3):
        x3 = self.proj(x3)
        out = self.drop(x3)
        return out


class SPBlock(nn.Module):
    def __init__(self, b, model_dim, p_size, p_num, drop_rate):
        super(SPBlock, self).__init__()
        self.b = b
        self.model_dim = model_dim
        self.p_num = p_num
        self.p_size = p_size
        self.sparse_proj = SparseProjection(64, 512, sparsity=0.4)
        self.down = nn.Linear(model_dim * 2, model_dim)
        self.dense = nn.Linear(model_dim * 2, model_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x4, training=True, m=None):
        b = x4.size(0)
        feat = x4.view(b, -1)
        feat = self.sparse_proj(feat)
        feat = feat.view(b, 2, 256)
        B = feat.shape[0] // self.p_num
        p_len = feat.shape[1]
        feat = feat.view(B, self.p_num, p_len, -1)
        p_feat = feat[:, :, -1, :]
        out1 = self.drop(self.down(p_feat))
        out2 = self.drop(self.dense(p_feat))

        return out1, out2