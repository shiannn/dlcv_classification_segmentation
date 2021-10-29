import torch
import torchvision
from config_p1 import LR, MOMENTUM, SCHEDULER_STEPSIZE, SCHEDULER_GAMMA
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

models = [
    #torchvision.models.vgg13(pretrained=True),
    #torchvision.models.vgg16(pretrained=True),
    #torchvision.models.vgg19(pretrained=True),
    #torchvision.models.resnet18(pretrained=True),
    #torchvision.models.vgg11_bn(pretrained=True),
    #torchvision.models.vgg13_bn(pretrained=True),
    torchvision.models.vgg16_bn(pretrained=True),
    torchvision.models.vgg19_bn(pretrained=True),
    #torchvision.models.resnet101(pretrained=True),
    #torchvision.models.resnet152(pretrained=True),
]
names = ['vgg16_bn', 'vgg19_bn']

criterion = torch.nn.CrossEntropyLoss()
optimizers = [
    torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM) for model in models
]
### Every 7 epoches decay lr with factor=0.1
"""
lr_schedulers = [
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEPSIZE, gamma=SCHEDULER_GAMMA) 
    for optimizer in optimizers
]
"""
lr_schedulers = [
    torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=LR,max_lr=1e-2,step_size_up=2000)
    for optimizer in optimizers
]
