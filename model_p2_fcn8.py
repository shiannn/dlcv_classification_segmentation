import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class SegmentationWithFCN8(nn.Module):
    def __init__(self, backbone, num_classes=7):
        super(SegmentationWithFCN8, self).__init__()
        #self.backbone = backbone
        #feats = list(backbone.children())
        feats = list(torchvision.models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:9])
        self.feat3 = nn.Sequential(*feats[10:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        del feats
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False
        
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 512, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.score_fconn = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat3 = self.feat3(feats)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.interpolate(score_fconn, score_feat4.shape[2:])
        score += score_feat4
        score = F.interpolate(score, score_feat3.shape[2:])
        score += score_feat3

        return F.interpolate(score, x.shape[2:])

if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16(pretrained=True)
    segmentationWithFCN8 = SegmentationWithFCN8(backbone=vgg16.features)
    tens = torch.rand((4,3,224,224))
    out = segmentationWithFCN8(tens)
    print(out.shape)