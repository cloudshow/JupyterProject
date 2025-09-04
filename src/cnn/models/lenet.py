import torch.nn as nn


class LeNet(nn.Module):
    """
        LeNet5 (Conv * 3 + Linear * 2)
        激活函数: Sigmoid
        池化: 平均池化
    """
    def __init__(self):
        super(LeNet, self).__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = LeNet()
    summary(model, input_size=(1, 28, 28), device='cpu')
