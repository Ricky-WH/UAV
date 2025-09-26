import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# 通道注意力（SE）模块
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // r, 1)
        self.fc2 = nn.Conv2d(channels // r, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

# 可变形卷积瓶颈（深度可分离 + 残差）
class MFC_Bottleneck(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # 偏移量生成器：2*k*k*groups
        self.offset = nn.Conv2d(
            channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=1,
            padding=dilation,
            bias=True,
        )
        # 可变形卷积（这里仍按通道分组即深度可分离）
        self.deform_conv = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        offset = self.offset(x)
        out = self.deform_conv(x, offset)
        out = self.act(self.bn(out))
        return out + identity

# 主模块
class MFC(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, dilation=1, se_r=16):
        """
        c1: 输入通道
        c2: 输出通道
        n: bottleneck 数量
        e: 中间通道缩放比
        dilation: 可变形卷积的空洞率
        se_r: SE 模块压缩比
        """
        super().__init__()
        n = int(n)
        hidden_channels = int(c2 * e)

        # 降维
        self.conv1 = nn.Conv2d(c1, hidden_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.SiLU(inplace=True)

        # n 个可变形卷积瓶颈
        self.bottlenecks = nn.ModuleList(
            [MFC_Bottleneck(hidden_channels, dilation=dilation) for _ in range(n)]
        )

        # 升维
        self.conv2 = nn.Conv2d(hidden_channels, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True)

        # SE 注意力
        self.se = SEBlock(c2, r=se_r)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x

# 测试
if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = MFC(64, 128, n=2, e=0.5, dilation=2, se_r=8)
    y = model(x)
    print(y.shape)  # torch.Size([1, 128, 32, 32])