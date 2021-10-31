import os
import torch
import numpy as np
from config_p2 import TRAIN_ROOT, VAL_ROOT, IMAGE_SIZE, DEVICE
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, common_transform=None, 
    transform=None, second_transform=None, target_transform=None):
        self.root_dir = root_dir
        datas = []
        sats, masks = [],[]
        for img_name in sorted(os.listdir(self.root_dir)):
            img_name_nosuffix = os.path.splitext(img_name)[0]
            img_id, sat_mask = img_name_nosuffix.split('_')
            if sat_mask == 'sat':
                sats.append(img_name)
            elif sat_mask == 'mask':
                masks.append(img_name)
        for sat, mask in zip(sats, masks):
            sat_id, sat_attach = os.path.splitext(sat)[0].split('_')
            mask_id, mask_attach = os.path.splitext(mask)[0].split('_')
            assert sat_id == mask_id and sat_attach == 'sat' and mask_attach == 'mask'
            sat_abs = os.path.join(self.root_dir, sat)
            mask_abs = os.path.join(self.root_dir, mask)
            datas.append((sat_abs, mask_abs))
        self.datas = datas
        self.common_transform = common_transform
        self.transform = transform
        self.second_transform = second_transform
        self.target_transform = target_transform
        if len(datas) > 0:
            self.ori_size = Image.open(datas[0][0]).convert('RGB').size
    
    def __getitem__(self, index):
        img_path, mask_path = self.datas[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        if self.common_transform is not None:
            #img = self.common_transform(img)
            #mask = self.common_transform(mask)
            img, mask = self.common_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        ### for other methods
        if self.second_transform is not None:
            img = self.second_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        ### turn onehot [3,244,244] into [244,244]
        mask = onehot2maskclass(mask)
        return img, mask

    def __len__(self):
        return len(self.datas)

def get_train_common_transform():
    """
    transform = transforms.Compose([
        #transforms.RandomCrop(size=32, padding=4),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
    ])
    return transform
    """
    def common_transform(img, mask, degree=15):
        img = torchvision.transforms.functional.resize(img, size=IMAGE_SIZE)
        mask = torchvision.transforms.functional.resize(mask, size=IMAGE_SIZE)
        ### random rotate
        rotate_rand = (-1)* degree + 2*degree* torch.rand(size=(1,))
        #print('rotate_rand', rotate_rand.item())
        img = torchvision.transforms.functional.rotate(img, angle=rotate_rand)
        mask = torchvision.transforms.functional.rotate(mask, angle=rotate_rand)
        ### random hflip
        flip_rand = torch.rand(size=(1,))
        #print('flip_rand', flip_rand.item())
        if flip_rand.item() > 0.5:
            img = torchvision.transforms.functional.hflip(img)
            mask = torchvision.transforms.functional.hflip(mask)
        return img, mask
    return common_transform

def get_train_transform(jitter_param=0.4):
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
    ])
    return transform

def get_train_target_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform
    
def onehot2maskclass(batch_masks):
    ### batch_masks [32, 3, 244, 244]
    #print(batch_masks.shape)
    #class_masks = 4*batch_masks[:, 0, :, :] + 2*batch_masks[:, 1, :, :] + 1*batch_masks[:, 2, :, :]
    class_masks = 4*batch_masks[0, :, :] + 2*batch_masks[1, :, :] + 1*batch_masks[2, :, :]
    #print(class_masks.shape)
    ### should simultaneously change these labels
    ### initialize another one
    class_masks_copy = torch.zeros_like(class_masks)#.to(DEVICE)
    class_masks_copy[class_masks == 3] = 0  # (Cyan: 011) Urban land 
    class_masks_copy[class_masks == 6] = 1  # (Yellow: 110) Agriculture land 
    class_masks_copy[class_masks == 5] = 2  # (Purple: 101) Rangeland 
    class_masks_copy[class_masks == 2] = 3  # (Green: 010) Forest land 
    class_masks_copy[class_masks == 1] = 4  # (Blue: 001) Water 
    class_masks_copy[class_masks == 7] = 5  # (White: 111) Barren land 
    class_masks_copy[class_masks == 0] = 6  # (Black: 000) Unknown 
    class_masks_copy = class_masks_copy.long()
    return class_masks_copy

def maskclass2onehot(batch_class):
    ### batch_class [IMAGE_NUM, 512, 512]
    ### first turn back the labels
    batch_class_copy = np.zeros_like(batch_class)
    batch_class_copy[batch_class==0] = 3
    batch_class_copy[batch_class==1] = 6
    batch_class_copy[batch_class==2] = 5
    batch_class_copy[batch_class==3] = 2
    batch_class_copy[batch_class==4] = 1
    batch_class_copy[batch_class==5] = 7
    batch_class_copy[batch_class==6] = 0
    print(batch_class_copy.shape)
    ret_onehot = np.zeros(
        (batch_class_copy.shape[0], batch_class_copy.shape[1], batch_class_copy.shape[2], 3)
    )
    ### turn into one-hot
    ret_onehot[:,:,:,0] = batch_class_copy
    ret_onehot[:,:,:,2] = ret_onehot[:,:,:,0] % 2
    ret_onehot[:,:,:,1] = (ret_onehot[:,:,:,0] // 2) % 2
    ret_onehot[:,:,:,0] = (ret_onehot[:,:,:,0] // 4) % 2
    #ret_onehot[1] = ret_onehot[1] % 2
    #ret_onehot = np.transpose(ret_onehot, (1,2,3,0))
    return ret_onehot
"""
def my_mean_iou_score(pred, labels):
    ### pred [32, 244, 244], labels [32, 244, 244] should be tensors
    ### only compute 0,1,2,3,4,5 except background
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = None
    for i in range(6):
        #print((pred == i).shape)
        #print((labels == i).shape)
        tp_fp = (pred == i).sum((2,1)).float()
        tp_fn = (labels == i).sum((2,1)).float()
        print(tp_fp)
        print(tp_fn)
        temp = ((pred == i) & (labels == i))
        tp = temp.sum((2,1)).float()
        print(tp)
        #tp = np.sum((pred == i) * (labels == i))
        #iou = tp / (tp_fp + tp_fn - tp)
        print(tp.type(), (tp_fp + tp_fn - tp).type())
        iou = torch.div(tp, (tp_fp + tp_fn - tp)+1e-5)
        print(iou)
        #mean_iou += iou / 6
        mean_iou = iou/6 if mean_iou is None else mean_iou + iou/6
        print(mean_iou)
        exit(0)
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
"""
def get_valid_common_transform():
    """
    transform = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
    ])
    return transform
    """
    def common_transform(img, mask, degree=15):
        img = torchvision.transforms.functional.resize(img, size=IMAGE_SIZE)
        mask = torchvision.transforms.functional.resize(mask, size=IMAGE_SIZE)
        return img, mask
    return common_transform

def get_valid_transform(jitter_param=0.4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
    ])
    return transform

def get_valid_target_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform

if __name__ == '__main__':
    mock_preds = np.load('mock_preds.npy')
    onehot_mask = maskclass2onehot(mock_preds)
    print(onehot_mask.shape)
    print(onehot_mask.max())
    print(onehot_mask.min())
    """
    print(mock_preds[0,0,0])
    print(onehot_mask[0,0,0,:])
    print(mock_preds[10,20,30])
    print(onehot_mask[10,20,30,:])
    print(mock_preds[70,50,80])
    print(onehot_mask[70,50,80,:])
    print(mock_preds[15,97,69])
    print(onehot_mask[15,97,69,:])
    print(mock_preds[127,55,45])
    print(onehot_mask[127,55,45,:])
    """
    """
    train_common_transform = get_train_common_transform()
    train_transform = get_train_transform(jitter_param=0.4)
    train_target_transform = get_train_target_transform()
    segmentationDataset = SegmentationDataset(
        TRAIN_ROOT, 
        common_transform=train_common_transform, 
        transform = train_transform,
        target_transform=train_target_transform
    )
    img, mask = segmentationDataset[57]
    #print(mask[:,345,450])
    #labels = 4*mask[2,:,:]+2*mask[1,:,:]+1*mask[0,:,:]
    #print(labels)
    loader = torch.utils.data.DataLoader(
        segmentationDataset, batch_size=32,
        shuffle=True, num_workers=4
    )
    distri = [0.0 for i in range(7)]
    for idx, batch_data in enumerate(loader):
        batch_imgs, batch_masks = batch_data
        total = batch_masks.shape[0]* batch_masks.shape[1]* batch_masks.shape[2]
        temp = 0
        for i in range(7):
            distri[i] += (batch_masks == i).sum()
            temp += (batch_masks == i).sum()
        assert total==temp
        if idx % 20 == 0:
            print(distri)
    print(distri)
    overall = sum([a.item() for a in distri])
    ratio = [overall / a for a in distri]
    print(ratio)
    adjust_ratio = [r / min(ratio) for r in ratio]
    print(adjust_ratio)
    """