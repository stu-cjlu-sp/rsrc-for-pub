import torch
import torch.nn as nn
from .blocks import OverlappingPatchSplitter, SPBlock
from .encoder import DualMambaEncoderBlock


class Model(nn.Module):
    def __init__(self, n_layers, model_dim, dff, num_class,
                 max_len_en, max_len, p_size, overlap, n_patch,
                 batch_size, drop_rate):
        super().__init__()
        self.num_class = num_class
        self.max_len = max_len
        self.patch_split = OverlappingPatchSplitter(max_len_en, p_size, overlap)
        self.patch_encoder = SPBlock(batch_size, model_dim, p_size, n_patch, drop_rate)
        self.encoder = DualMambaEncoderBlock(n_layers, model_dim, dff, max_len_en, drop_rate)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(model_dim*2, model_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(model_dim*2, model_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        self.fnn1 = nn.Sequential(nn.Linear(model_dim * 2, model_dim * 2),nn.GELU(),
                                  nn.Dropout(drop_rate),nn.Linear(model_dim * 2, model_dim * 2))
        self.fnn2 = nn.Sequential(nn.Linear(model_dim * 2, model_dim * 2),nn.GELU(),
                                  nn.Dropout(drop_rate),nn.Linear(model_dim * 2, model_dim * 2))
        self.head1 = nn.Linear(model_dim * 2, num_class * max_len)
        self.head2 = nn.Linear(model_dim * 2, max_len * 2)
        self.out1 = nn.Linear(num_class, num_class)
        self.out2 = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x5, training):
        patch_feat = self.patch_split(x5)
        patch_mask = pad_mask(patch_feat)
        feat1, feat2 = self.patch_encoder(x4=patch_feat, training=training, m=patch_mask)
        mask = pad_mask(feat1[:, :, 0] + 1)
        enc_out = self.encoder(feat1, feat2, mask)

        enc_out_rs = enc_out.permute(0, 2, 1)
        
        pooled = self.pool(enc_out_rs).squeeze(-1)
        
        f1 = self.fc1(pooled)
        f2 = self.fc2(pooled)

        ff1 = self.fnn1(f1)
        logits = self.head1(ff1)
        logits = logits.view(-1, self.max_len, self.num_class)
        out1 = self.out1(logits)

        
        ff2 = self.fnn2(f2)
        reg = self.head2(ff2)
        reg = reg.view(-1, self.max_len, 2)
        out2 = self.sigmoid(self.out2(reg))

        return out1, out2, logits