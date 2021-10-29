import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from config_p2 import LR, MOMENTUM

class SegmentationWithFCN32(torch.nn.Module):
    def __init__(self, backbone, num_classes=7):
        super(SegmentationWithFCN32, self).__init__()
        self.backbone = backbone
        self.fcnn = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1)
        self.convTranspose = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32)
    
    def forward(self, x):
        feats = self.backbone(x)
        #print(feats.shape)
        fconn = self.fcnn(feats)
        #print(fconn.shape)
        score = self.score(fconn)
        #return F.interpolate(score, x.shape[2:])
        #print(score.shape)
        return self.convTranspose(score)

def get_criterion():
    criterion = torch.nn.CrossEntropyLoss()
    return criterion
def get_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    return optimizer
def get_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,base_lr=LR,max_lr=1e-2,step_size_up=2000
    )
    return scheduler

if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16(pretrained=True)
    segmentationWithFCN32 = SegmentationWithFCN32(backbone=vgg16.features)
    tens = torch.rand((4,3,224,224))
    out = segmentationWithFCN32(tens)
    print(out.shape)