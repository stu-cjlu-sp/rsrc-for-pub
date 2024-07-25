import torch.nn as nn
import torch
from complexLayers import ComplexConv1d

def set_layer1_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
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


import math
class convolutional_layers(nn.Module):
    def __init__(self, num_layers=8, num_features=8):
        super(convolutional_layers, self).__init__()
        self.num_layers = num_layers

        conv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))


        self.conv_layers = nn.Sequential(*conv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        clax=x

        return x,clax,conv_feats
       
class deconvolutional_layers(nn.Module):
    def __init__(self, num_layers=8, num_features=8):
        super(deconvolutional_layers, self).__init__()
        self.num_layers = num_layers

        deconv_layers = []

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1,conv_feats):

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x1 = self.deconv_layers[i](x1)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x1 = x1 + conv_feat
                x1 = self.relu(x1)

        x1 += x
        x1 = self.relu(x1)

        return x1

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
        

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(32, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            feature = torch.flatten(x, 1)
            x = self.fc(feature)

        return x

class FrequencyRepresentationModule_TFA_Net(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.n_filters = n_filters
        self.inner = inner_dim
        self.n_layers = n_layers
        num_classes=12
        include_top=True

        self.in_layer1 = ComplexConv1d(1, inner_dim * n_filters, kernel_size=(1, 31), padding=(0, 31 // 2),
                                       bias=False)

        self.rednetcode=convolutional_layers(self.n_layers,num_features=n_filters)
        self.rednetdecode=deconvolutional_layers(self.n_layers,num_features=n_filters)
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (3, 1), stride=(upsampling, 1),
                                            padding=(1, 0), output_padding=(1, 0), bias=False)


    def forward(self, x):
        bsz = x.size(0) 
        inp_real = x[:, 0, :].view(bsz, 1, 1, -1) 
        inp_imag = x[:, 1, :].view(bsz, 1, 1, -1)
        inp = torch.cat((inp_real, inp_imag), 1) 
        x1 = self.in_layer1(inp) 
        
        # Modules
        xreal,ximag=torch.chunk(x1,2,1) 
        xreal = xreal.view(bsz, self.n_filters, self.inner, -1) 
        ximag = ximag.view(bsz, self.n_filters, self.inner, -1)

        x=torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2)) 
        x1,clax,conv_feats=self.rednetcode(x) 
        output = self.resnet(clax)
        x2 =self.rednetdecode(x, x1, conv_feats)
        x3 = self.out_layer(x2) 
        ximg = x3.squeeze(-3)
        return ximg, output,x,clax
  