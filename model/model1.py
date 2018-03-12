import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    name = 'Model1'

    def __init__(self, category_size, phase='train'):
        super(Model1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=24 * 13 * 13, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=category_size),
        )

        self.phase = phase

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size()[0], -1)
        x = self.fc_layer(x)
        return x
