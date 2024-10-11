import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, configs, depth, width_factor, num_classes=11):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'Depth must be 6n+4'
        n = (depth - 4) // 6
        k = width_factor

        n_stages = [16, 16 * k,  32*k, 64*k]

        self.conv1 = nn.Conv2d(9, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._wide_stage(n_stages[0], n_stages[1], n, stride=1)
        self.stage2 = self._wide_stage(n_stages[1], n_stages[2], n, stride=2)
        self.stage3 = self._wide_stage(n_stages[2], n_stages[3], n, stride=2)
        self.bn = nn.BatchNorm2d(n_stages[3])
        self.fc = nn.Linear(n_stages[3], num_classes)

    def _wide_stage(self, in_channels, out_channels, n, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out