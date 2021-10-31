import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class SegmentationWithFCN16(nn.Module):
    def __init__(self, num_classes=7):
        super(SegmentationWithFCN16, self).__init__()
        feats = list(torchvision.models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        del feats
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 1024, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(1024, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.interpolate(score_fconn, score_feat4.shape[2:])
        score += score_feat4

        return F.interpolate(score, x.shape[2:])

if __name__ == '__main__':
    #vgg16 = torchvision.models.vgg16(pretrained=True)
    segmentationWithFCN16 = SegmentationWithFCN16(num_classes=7)
    tens = torch.rand((4,3,224,224))
    out = segmentationWithFCN16(tens)
    print(out.shape)