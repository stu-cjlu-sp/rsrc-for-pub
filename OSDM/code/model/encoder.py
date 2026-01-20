import torch
import torch.nn as nn
from .blocks import Mambalayer


class DualMambaEncoderBlock(nn.Module):
    def __init__(self, n_layers, model_dim, ddf, max_len, drop_rate):
        super().__init__()

        self.n_layers = n_layers
        self.model_dim = model_dim

        self.proj1 = nn.Linear(model_dim, model_dim)
        self.proj2 = nn.Linear(model_dim, model_dim)
        self.register_buffer('pos_emb',
                             self._get_pos_enc(max_len, model_dim))
        self.layers1 = nn.ModuleList([Mambalayer(model_dim, ddf, drop_rate)
                                             for _ in range(n_layers)])
        self.layers2 = nn.ModuleList([Mambalayer(model_dim, ddf, drop_rate)
                                             for _ in range(n_layers)])
        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

    def _get_pos_enc(self, max_len, model_dim):
        pos = torch.arange(0, max_len).unsqueeze(1)
        scale  = torch.exp(torch.arange(0, model_dim, 2) *
                             -(torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(1, max_len, model_dim)
        pe[0, :, 0::2] = torch.sin(pos * scale )
        pe[0, :, 1::2] = torch.cos(pos * scale )
        return pe

    def forward(self, x1, x2, mask=None):
        len1 = x1.size(1)
        len2 = x2.size(1)

        h1 = self.proj1(x1)
        h2 = self.proj2(x2)

        pe1 = self.pos_emb[:, :len1, :].to(x1.device)
        pe2 = self.pos_emb[:, :len2, :].to(x2.device)

        feat1 = h1 + pe1
        feat2 = h2 + pe2
        feat1 = self.drop1(feat1)
        feat2 = self.drop2(feat2)

        for layer in self.layers1:
            feat1 = layer(feat1, mask)
        for layer in self.layers2:
            feat2 = layer(feat2, mask)
        out3 = torch.cat([feat1, feat2], dim=2)
        return out3