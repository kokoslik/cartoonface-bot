from torch import nn


class Discriminator(nn.Module):
    def __init__(self, instance_norm=False):
        super().__init__()
        NormLayer = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=instance_norm),
            NormLayer(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,128, kernel_size=4, stride=2, padding=1, bias=instance_norm),
            NormLayer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=instance_norm),
            NormLayer(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=instance_norm),
            NormLayer(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)
