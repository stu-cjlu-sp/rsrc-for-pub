import torch
import torch.nn as nn
import torch.nn.functional as F



class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo



def custom_relu(x, threshold):
    return torch.where(x > threshold, x, torch.tensor(0.0, device=x.device))

class Denoise(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(Denoise, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1 ,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.excitation = nn.Sequential(
            nn.Linear(out_channel, out_channel//4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel//4, out_channel),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(out_channel, out_channel*2, kernel_size=3, stride=2, padding=1),  # 输入通道数为3（RGB图像）
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel*2, out_channel*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel*4, out_channel*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channel*2, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数将输出限制在 [0, 1] 范围内
        )

    # def forward(self, x):
    #     x = self.conv(x)
    #     x = self.mp(x)
    #     x_G = F.adaptive_avg_pool2d(x, (1, 1))
    #     x_G = torch.flatten(x_G, 1)
    #     x_SE = self.excitation(x_G)
    #     x_SE = x_SE.unsqueeze(2).unsqueeze(2)
    #     threshold = x*x_SE
    #     result = custom_relu(x, threshold)
    #     return result

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.mp(x1)
        x2 = self.encoder(x1)
        x2 = self.decoder(x2)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        result = custom_relu(x1, x2)
        return result

class Attentional(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(Attentional , self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1 ,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=5, stride=1, padding=2 ,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.AFF = AFF(out_channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x_SE = self.conv2(x)
        # x_G = F.adaptive_avg_pool2d(x, (1, 1))
        # x_G = torch.flatten(x_G, 1)
        # x_SE = self.excitation(x_G)
        result = self.AFF(x1, x_SE)
        return result

class CNN(nn.Module):
    def __init__(self,in_dim) -> None:
        super(CNN, self).__init__()
        self.conv1 = Denoise(in_dim,8)
        self.AFF1 = Attentional(8,8)
        self.conv2 = Denoise(8,16)
        self.AFF2 = Attentional(16,16)
        self.conv3 = Denoise(16,32)
        self.AFF3 = Attentional(16,16)
        self.prob = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        self.thre = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
        self.sfa = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8,kernel_size=2, stride=1),
            nn.ReLU(inplace=True)
        )
        self.sfa2 = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.AFF1(x)
        x = self.conv2(x)
        x = self.AFF2(x)

        x_thre = self.AFF3(x)
        x_thre = torch.flatten(x_thre, 1)
        x_thre = self.sfa2(x_thre)

        # x_prob = self.sfa(x)
        # x_prob = torch.flatten(x_prob, 1)
        # x_prob = self.sfa2(x_prob)

        x = torch.flatten(x, 1)
        x_prob = self.prob(x)
        # x_thre = self.thre(x)

        return x_prob, x_thre
        # return x_prob

if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"

    # [batch,channel,H,W]
    img = torch.rand(16, 1, 224, 224).cuda()
    model = CNN(1).cuda()
    result,result2 = model(img)
    print(result.shape)