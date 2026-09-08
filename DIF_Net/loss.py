import torch
import torch.nn as nn

class UncertaintyWeightLoss(nn.Module):
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    def __init__(self):
        super().__init__()
        self.log_var_mod = nn.Parameter(torch.zeros(1))
        self.log_var_fc = nn.Parameter(torch.zeros(1))
        self.log_var_spec = nn.Parameter(torch.tensor(init_spec_logvar))

    def forward(self, loss_mod, loss_fc, loss_spec):
        # 分类 loss（CE）
        loss_mod_weighted = torch.exp(-self.log_var_mod) * loss_mod + self.log_var_mod

        # 回归 loss（MSE）
        loss_fc_weighted = 0.5 * torch.exp(-self.log_var_fc) * loss_fc + 0.5 * self.log_var_fc
        loss_spec_weighted = 0.5 * torch.exp(-self.log_var_spec) * loss_spec + 0.5 * self.log_var_spec

        total_loss = loss_mod_weighted + loss_fc_weighted + loss_spec_weighted

        return total_loss, loss_mod_weighted, loss_fc_weighted, loss_spec_weighted
