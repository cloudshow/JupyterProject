import torch.nn as nn
import torchsummary


class CBL(nn.Module):
    """
    CBL Module: Convolution + BatchNorm + Leaky ReLU Activation
    Conv2d层 bias=False: 在使用了批量归一化后通常不需要偏置, BatchNorm 层本身有可学习的偏置参数
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CBL, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ResUnit(nn.Module):
    """
    Residual Unit: CBL + CBL + Residual Connection
    CBL1: 1x1 点卷积降低通道数 1/2
    CBL2: 3x3 卷积提升回通道数 * 2
    Residual Connection
    """

    def __init__(self, in_channels):
        super(ResUnit, self).__init__()
        assert (
            in_channels % 2 == 0
        ), f"in_channels must be divisible by 2, but got {in_channels}"
        self.net = nn.Sequential(
            CBL(in_channels, in_channels // 2, kernel_size=1),
            CBL(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x
        return residual + self.net(x)


class ResX(nn.Module):
    """
    Residual X: CBL + Residual Unit * x
    """

    def __init__(self, in_channels, num_res_units):
        super(ResX, self).__init__()
        """
        modules = [
            CBL(in_channels, in_channels * 2, 3, stride=2, padding=1)  # Downsample
        ]
        for _ in range(num_res_units):
            modules.append(ResUnit(in_channels * 2))
        self.net = nn.Sequential(*modules) 
        """
        self.net = nn.Sequential(
            CBL(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
            *[ResUnit(in_channels * 2) for _ in range(num_res_units)],
        )

    def forward(self, x):
        return self.net(x)


class ConvSet(nn.Module):
    def __init__(self, in_channels):
        super(ConvSet, self).__init__()
        self.net = nn.Sequential(
            CBL(in_channels=512, out_channels=256, kernel_size=1),
            CBL(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CBL(in_channels=512, out_channels=256, kernel_size=1),
            CBL(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CBL(in_channels=512, out_channels=256, kernel_size=1)
        )

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # 初始层
        self.stem = CBL(3, 32, 3, 1, 1)
        # 主干残差块
        self.layer1 = ResX(in_channels=32, num_res_units=1)
        self.layer2 = ResX(in_channels=64, num_res_units=2)
        self.layer3 = ResX(in_channels=128, num_res_units=8)
        self.layer4 = ResX(in_channels=256, num_res_units=8)
        self.layer5 = ResX(in_channels=512, num_res_units=4)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        fm52 = self.layer3(x)
        fm26 = self.layer4(fm52)
        fm13 = self.layer5(fm26)
        return fm52, fm26, fm13


if __name__ == "__main__":
    from torchsummary import summary
    import torch
    # Backbone
    model = Backbone()
    fm52, fm26, fm32 = model(torch.randn(1, 3, 416, 416))
    print(fm52.shape, fm26.shape, fm32.shape)
    summary(model, input_size=(3, 416, 416), batch_size=1, device="cpu")
    # Neck
