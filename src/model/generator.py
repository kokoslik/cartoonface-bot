from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, instance_norm=False, dropout=False):
        super().__init__()
        NormLayer = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
	
        if dropout:
            self.convBlock = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=instance_norm),
                NormLayer(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=instance_norm),
                NormLayer(dim)
            )
        else:
            self.convBlock = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=instance_norm),
                NormLayer(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=instance_norm),
                NormLayer(dim)
            )

    def forward(self, x):
        return x + self.convBlock(x)


class Generator(nn.Module):
    def __init__(self, instance_norm=False, dropout=False):
        super().__init__()
        NormLayer = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=instance_norm), #256x256
            NormLayer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=instance_norm), #128x128
            NormLayer(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=instance_norm), #64x64
            NormLayer(256),
            nn.ReLU(True),
        )
        self.resnet = nn.Sequential(*[ResidualBlock(256,instance_norm, dropout) for i in range(6)])
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=instance_norm), #128x128
            NormLayer(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=instance_norm), #256x256
            NormLayer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=True), #256x256
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = self.resnet(x)
        x = self.encoder(x)
        return x
