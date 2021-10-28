import torch
import torchvision
from config_p1 import LR, MOMENTUM, SCHEDULER_STEPSIZE, SCHEDULER_GAMMA
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = torchvision.models.vgg16(pretrained=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
### Every 7 epoches decay lr with factor=0.1
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEPSIZE, gamma=SCHEDULER_GAMMA)