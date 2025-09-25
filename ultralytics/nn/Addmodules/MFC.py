import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MFC']

# 一个基础的 Bottleneck，可以根据需要改进
class MFC_Bottleneck(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        # 深度可分离卷积
        self.dw = nn.Conv2d(channels, channels, 3, 1, dilation=dilation,
                            padding=dilation, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn(self.dw(x)))
        return out + identity


class MFC(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        n = int(n)   # 🔑 确保 n 是整数
        hidden_channels = int(c2 * e)

        self.conv1 = nn.Conv2d(c1, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.bottlenecks = nn.ModuleList([MFC_Bottleneck(hidden_channels) for _ in range(n)])

        self.conv2 = nn.Conv2d(hidden_channels, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        x = self.act2(self.bn2(self.conv2(x)))
        return x