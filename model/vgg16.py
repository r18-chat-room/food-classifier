import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    """
    VGG with 16 layers.

    ConvNet Configuration given as : conv<receptive field size>-<number of channels>.

    conv3-16  -> conv3-16  -> maxpool   ->
    conv3-32  -> conv3-32  -> maxpool   ->
    conv3-64 -> conv3-64 -> conv3-64 -> maxpool ->
    conv3-128 -> conv3-128 -> conv3-128 -> maxpool ->
    conv3-128 -> conv3-128 -> conv3-128 -> maxpool ->
    fc-512 -> fc-512 -> fc-<category_size>

    """
    name = 'VGG-16'

    def __init__(self, category_size, phase='train'):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 7 * 7, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=category_size),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
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
