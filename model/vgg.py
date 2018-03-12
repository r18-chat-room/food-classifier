import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    """
    variant from http://torch.ch/blog/2015/07/30/cifar.html
    """
    name = 'VGG'

    def __init__(self, category_size, phase='train'):
        super(VGG, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBNReLU(3, 64),
            nn.Dropout2d(0.3),
            ConvBNReLU(64, 64),
        )
        self.conv2 = nn.Sequential(
            ConvBNReLU(64, 128),
            nn.Dropout2d(0.4),
            ConvBNReLU(128, 128),
        )
        self.conv3 = nn.Sequential(
            ConvBNReLU(128, 256),
            nn.Dropout2d(0.4),
            ConvBNReLU(256, 256),
            nn.Dropout2d(0.4),
            ConvBNReLU(256, 256),
        )
        self.conv4 = nn.Sequential(
            ConvBNReLU(256, 512),
            nn.Dropout2d(0.4),
            ConvBNReLU(512, 512),
            nn.Dropout2d(0.4),
            ConvBNReLU(512, 512),
        )
        self.conv5 = nn.Sequential(
            ConvBNReLU(512, 512),
            nn.Dropout2d(0.4),
            ConvBNReLU(512, 512),
            nn.Dropout2d(0.4),
            ConvBNReLU(512, 512),
        )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=category_size),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.phase = phase

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.maxpool(self.conv4(x))
        x = self.maxpool(self.conv5(x))     # features
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBNReLU, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(num_features=out_channels, eps=10**(-3)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
