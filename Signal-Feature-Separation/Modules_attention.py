import torch.nn as nn
import torch
from complexLayers import ComplexConv1d
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os
import math



def set_layer1_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling//2, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_TFA_Net(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)


    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

    
class REDNet30_stft(nn.Module):
    def __init__(self, num_layers=8, num_features=8):
        super(REDNet30_stft, self).__init__()
        self.num_layers = num_layers
        self.channelattention = ChannelAttention(num_features*2)
        self.spatialattention = SpatialAttention()

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + self.spatialattention(self.channelattention(conv_feat))
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class Conv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Conv, self).__init__()

        self.channelattention = ChannelAttention(out_dim)
        self.spatialattention = SpatialAttention()

        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.MP = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = self.conv1(x) 
        x_attention = self.channelattention(x)
        x_attention = self.spatialattention(x_attention)
        x_out = self.MP(x)

        return x_out, x_attention


class gesture_segmentation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(gesture_segmentation, self).__init__()

        self.channelattention = ChannelAttention(512)
        self.spatialattention = SpatialAttention()

        self.encoder1 = Conv(in_dim,64)
        self.encoder2 = Conv(64,128)
        self.encoder3 = Conv(128,256)
        self.encoder4 = torch.nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.encoder4_mp = nn.MaxPool2d(2,2)

        self.decoder1 = torch.nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder2 = torch.nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder3 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder4 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.outlayer = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_dim, kernel_size=(3, 3),padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    def forward(self,x):
        x, x_attention1 = self.encoder1(x) 
        x, x_attention2 = self.encoder2(x) 
        x, x_attention3 = self.encoder3(x) 
        x4 = self.encoder4(x) 
        x4_channelattention = self.channelattention(x4)
        x_attention4 = self.spatialattention(x4_channelattention) 
        encoder4 = self.encoder4_mp(x4) 
        decoder1_output = self.decoder1(encoder4) 
        decoder1_attention = decoder1_output + x_attention4
        decoder2_output = self.decoder2(decoder1_attention) 
        decoder2_attention = decoder2_output + x_attention3
        decoder3_output = self.decoder3(decoder2_attention) 
        decoder3_attention = decoder3_output + x_attention2
        decoder4_output = self.decoder4(decoder3_attention)
        decoder4_attention = decoder4_output + x_attention1
        x = self.outlayer(decoder4_attention)

        return x
            



class FrequencyRepresentationModule_TFA_Net(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=128,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.n_filters = n_filters
        self.inner = inner_dim
        self.n_layers = n_layers
        include_top=True

        self.in_layer1 = ComplexConv1d(1, inner_dim * n_filters, kernel_size=(1, 31), padding=(0, 31 // 2),
                                       bias=False)

        self.rednetcode=gesture_segmentation(16,2)



    def forward(self, x):
        bsz = x.size(0) 
        inp_real = x[:, 0, :].view(bsz, 1, 1, -1) 
        inp_imag = x[:, 1, :].view(bsz, 1, 1, -1)
        inp = torch.cat((inp_real, inp_imag), 1) 
        x1 = self.in_layer1(inp) 
        
        xreal,ximag=torch.chunk(x1,2,1) 
        xreal = xreal.view(bsz, self.n_filters, self.inner, -1) 
        ximag = ximag.view(bsz, self.n_filters, self.inner, -1)

        x=torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2)) 




        img = self.rednetcode(x) 
        img1 = img[:, 0, :,:]
        img2 = img[:, 1, :,:]
        return img1, img2
    