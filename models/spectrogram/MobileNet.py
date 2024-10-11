import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                        groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, configs, num_classes=11):
        super(MobileNetV2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),

            # DepthwiseSeparableConv(256, 512, 2),
            # DepthwiseSeparableConv(512, 512, 1),
            # DepthwiseSeparableConv(512, 512, 1),
            # DepthwiseSeparableConv(512, 512, 1),
            # DepthwiseSeparableConv(512, 512, 1),
            #
            # DepthwiseSeparableConv(512, 1024, 2),
            # DepthwiseSeparableConv(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


