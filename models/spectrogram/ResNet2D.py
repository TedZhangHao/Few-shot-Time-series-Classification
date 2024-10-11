import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1,1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        # Residual Block
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # in-situ replacement, reduce memory consumption
            nn.Conv2d(out_channels, out_channels,  kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut section
        # Due to the inconsistency in dimensions, situation should be handled on a case-by-case basis
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Using 1x1 convolution for dimensionality adjustment
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet2D(nn.Module):
    def __init__(self, configs) -> None:
        super(ResNet2D, self).__init__()
        self.in_channels = 16
        # The first layer does not contain residual block
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # dilation = 1
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, self.in_channels, [[1, 1], [1, 1]])
        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 32, [[2, 1], [1, 1]])
        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(32,4)

    # Create Residual Block
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        #out = self.conv4(out)
        #out = self.conv5(out)
        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out