import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationWithUNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SegmentationWithUNet, self).__init__()
        self.dec1 = UNetDecoder(3, 64)
        self.dec2 = UNetDecoder(64, 128)
        self.dec3 = UNetDecoder(128, 256)
        self.dec4 = UNetDecoder(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            #nn.Upsample(scale_factor=2),
            nn.ReLU(inplace=True),
        )
        self.enc4 = UNetEncoder(1024, 512, 256)
        self.enc3 = UNetEncoder(512, 256, 128)
        self.enc2 = UNetEncoder(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        d1 = self.dec1(x)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        center = self.center(d4)
        e4 = self.enc4(
            torch.cat([center, F.interpolate(d4, center.shape[2:])], 1)
        )
        e3 = self.enc3(
            torch.cat([e4, F.interpolate(d3, e4.shape[2:])], 1)
        )
        e2 = self.enc2(
            torch.cat([e3, F.interpolate(d2, e3.shape[2:])], 1)
        )
        e1 = self.enc1(
            torch.cat([e2, F.interpolate(d1, e2.shape[2:])], 1)
        )

        return F.interpolate(self.score(e1), x.shape[2:])

class UNetEncoder(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
            #nn.Upsample(scale_factor=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(.5))
        layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

if __name__ == '__main__':
    model = SegmentationWithUNet(num_classes=7)
    tens = torch.rand((4,3,224,224))
    out = model(tens)
    print(out.shape)