import torch
import torch.nn as nn

__all__ = ['HGF']

class GlobalEnhance(nn.Module):
    """轻量化全局增强: Depthwise Conv + Pointwise Conv"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)  # depthwise
        self.pwconv = nn.Conv2d(dim, dim, 1, bias=False)  # pointwise
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class FusionGate(nn.Module):
    """动态融合门控，预测两个路径的融合权重"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,1,1)
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 2, 1, bias=False),  # 输出两个权重
            nn.Softmax(dim=1)
        )

    def forward(self, xa, xb):
        # xa, xb: 两个路径的特征
        fuse_in = xa + xb
        w = self.fc(fuse_in)  # (B,2,1,1)
        wa, wb = w[:, 0:1], w[:, 1:2]
        return wa * xa + wb * xb


class HGF_Bottleneck_DualRoute(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = nn.Sequential(
            Partial_conv3(dim, n_div=4, forward='split_cat'),
            Partial_conv3(dim, n_div=4, forward='split_cat')
        )
        self.global_enh = GlobalEnhance(dim)
        self.gate = FusionGate(dim)

    def forward(self, x):
        xa = self.local(x)       # 局部增强
        xb = self.global_enh(x)  # 全局增强
        return self.gate(xa, xb) # 动态融合


class HGF_DualRoute(nn.Module):
    """创新版 LCF: 双路特征增强 + 动态门控融合"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(LCF_Bottleneck_DualRoute(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    image = torch.rand(1, 64, 224, 224)
    model = LCF_DualRoute(64, 128, n=2)
    out = model(image)
    print(out.size())  # torch.Size([1, 128, 224, 224])