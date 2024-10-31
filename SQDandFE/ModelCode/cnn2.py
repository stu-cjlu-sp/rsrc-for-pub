import torch
import numpy as np
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,in_dim,n_class) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=20, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=12, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(8),
            # nn.ReLU(True),
            # nn.MaxPool2d(2,2),
        )
        self.fc = nn.Sequential(
            nn.Linear(5000,128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128,n_class)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(84,n_class)
        )


    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  
        # x = self.fc1(x)
        x = self.fc(x)
        return x
    
class EnsembleModel1(nn.Module):
    def __init__(self) -> None:
        super(EnsembleModel1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3,6),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(6,3)
        )
    def forward(self, x1,x2,x3):
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x3 = self.fc(x3)
        x = x1 + x2 + x3
        return x
        
class EnsembleModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=6, output_dim=3) -> None:
        super(EnsembleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()

    def forward(self, x1, x2, x3):
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x3 = self.fc(x3)
        x = x1 + x2 + x3
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__=='__main__':
    x=torch.randn(8,1,224,224)
    model = CNN(1,12)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数数量为：{total_params}")
    # model = model.cuda()
    y = model(x)
    # z = softmax(y)
    print(y)
    print(y.shape)