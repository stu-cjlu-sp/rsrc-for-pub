import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models


class train_fc(nn.Module):
    def __init__(self):
        super(train_fc,self).__init__()
        self.fc= nn.Sequential(
            nn.Linear(65536,256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256,6),
            nn.BatchNorm1d(6)
        )

    def forward(self,x):
        x = x.view(x.size(0), -1)   #8,6728
        x = self.fc(x)

        return x

