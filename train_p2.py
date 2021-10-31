import os
import math
import torch
import torchvision
import numpy as np
from config_p2 import TRAIN_ROOT, VAL_ROOT, BATCH_SIZE, EPOCH, NUM_WORKERS, DEVICE, NUM_CLASSES, LR, MOMENTUM, WEIGHT, SAVE_DIR, MAX_MODEL_NUM
from dataset_p2 import (
    get_train_common_transform, get_train_transform, get_train_target_transform, 
    get_valid_common_transform, get_valid_transform, get_valid_target_transform,
    SegmentationDataset#, onehot2maskclass
)
from model_p2 import SegmentationWithFCN32#, get_criterion, get_optimizer, get_scheduler
from model_p2_unet import SegmentationWithUNet
from model_p2_fcn8 import SegmentationWithFCN8
from model_p2_fcn16 import SegmentationWithFCN16
from mean_iou_evaluate import mean_iou_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", type=str, choices=['unet', 'vgg-fcn32', 'vgg16bn-fcn32', 'vgg-fcn16', 'vgg-fcn8'])

    args = parser.parse_args()
    return args

def training(args):
    train_common_transform = get_train_common_transform()
    train_transform = get_train_transform(jitter_param=0.4)
    train_target_transform = get_train_target_transform()
    train_dataset = SegmentationDataset(
        TRAIN_ROOT, 
        common_transform=train_common_transform, 
        transform = train_transform,
        target_transform=train_target_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_common_transform = get_valid_common_transform()
    valid_transform = get_valid_transform()
    valid_target_transform = get_valid_target_transform()
    valid_dataset = SegmentationDataset(
        VAL_ROOT, 
        common_transform=valid_common_transform, 
        transform = valid_transform,
        target_transform=valid_target_transform
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    if args.architecture == 'unet':
        model = SegmentationWithUNet(num_classes=NUM_CLASSES)
    elif args.architecture == 'vgg-fcn32':
        vgg16 = torchvision.models.vgg16(pretrained=True)
        model = SegmentationWithFCN32(backbone=vgg16.features, num_classes=NUM_CLASSES)
    elif args.architecture == 'vgg16bn-fcn32':
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        model = SegmentationWithFCN32(backbone=vgg16_bn.features, num_classes=NUM_CLASSES)
    elif args.architecture == 'vgg-fcn16':
        #vgg16 = torchvision.models.vgg16(pretrained=True)
        model = SegmentationWithFCN16(num_classes=NUM_CLASSES)
    elif args.architecture == 'vgg-fcn8':
        vgg16 = torchvision.models.vgg16(pretrained=True)
        model = SegmentationWithFCN8(backbone=vgg16.features, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    """
    criterion = get_criterion()
    optimizer = get_optimizer(segmentationWithFCN32)
    lr_scheduler = get_scheduler(optimizer)
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,base_lr=LR,max_lr=1e-2,step_size_up=2000
    )
    for epoch in range(EPOCH):
        best_iou = -1.0
        for phase in ['train', 'val']:
            if phase == 'train':
                print('[train]')
                model.train()
                loader = train_loader
            else:
                print('[valid]')
                model.eval()
                loader = valid_loader
            running_loss = 0.0
            ### accumulate pred & labels
            preds = None
            labels = None
            for idx, batch_data in enumerate(loader):
                optimizer.zero_grad()
                batch_imgs, batch_masks = batch_data
                batch_imgs = batch_imgs.to(DEVICE)
                batch_masks = batch_masks.to(DEVICE)
                #class_masks = onehot2maskclass(batch_masks)
                class_masks = batch_masks
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(batch_imgs)
                    #print(output.shape)
                    #print(class_masks.shape)
                    ### output [32, 7, 244, 244] class_masks [32, 244, 244]
                    _, pred = torch.max(output, 1)
                    output = output.permute(0, 2, 3, 1)
                    output = output.reshape(-1, output.shape[3])
                    class_masks_1d = class_masks.reshape(-1)
                    #print(class_masks.min(), class_masks.max())
                    pixel_loss = criterion(output, class_masks_1d)
                    #print(pixel_loss)
                    if phase == 'train':
                        pixel_loss.backward()
                        optimizer.step()
                running_loss += pixel_loss.item()* batch_imgs.shape[0]
                preds = pred.cpu().numpy() if preds is None else np.concatenate((preds, pred.cpu().numpy()), axis=0)
                labels = class_masks.cpu().numpy() if labels is None else np.concatenate((labels, class_masks.cpu().numpy()), axis=0)
                
                
            mean_iou = mean_iou_score(preds, labels)
            if phase == 'train':
                lr_scheduler.step()
            if phase == 'train':
                epoch_loss = running_loss / len(train_dataset)
                #epoch_acc = running_corrects.double() / len(train_dataset)
            else:
                epoch_loss = running_loss / len(valid_dataset)
                #epoch_acc = running_corrects.double() / len(valid_dataset)
            print('Epoch:{} {} Loss: {:.4f} Mean_IOU: {:.4f}'.format(epoch, phase, epoch_loss, mean_iou))
            if phase == 'val' and mean_iou > best_iou:
                print('Saving model...')
                best_iou = mean_iou
                name_iou = str(math.floor(100*best_iou))
                name = args.architecture
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, name+'_'+name_iou+'.pkl'))
                ### remove unnecessary models
                saved_models = os.listdir(SAVE_DIR)
                if len(saved_models) > MAX_MODEL_NUM:
                    min_acc = None
                    model_to_removed = None
                    for saved_model in saved_models:
                        acc = os.path.splitext(saved_model)[0].split('_')[1]
                        acc = int(acc)
                        if min_acc is None or acc < min_acc:
                            min_acc = acc
                            model_to_removed = saved_model
                    print('Removing model {}'.format(model_to_removed))
                    os.remove(os.path.join(SAVE_DIR, model_to_removed))

if __name__ == '__main__':
    args = parse_args()
    training(args)