import os
import torch
import torch.nn.functional as F
import argparse
import torchvision
import numpy as np
from PIL import Image
from model_p2 import SegmentationWithFCN32
from model_p2_fcn16 import SegmentationWithFCN16
from config_p2 import BATCH_SIZE, NUM_WORKERS, DEVICE, NUM_CLASSES, SAVE_DIR
from dataset_p2 import (
    get_valid_common_transform, get_valid_transform, get_valid_target_transform,
    SegmentationDataset, maskclass2onehot
)
from mean_iou_evaluate import mean_iou_score

### The script is totally for testing, which has no groundtruth in the folder of images
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path for predicting")
    parser.add_argument("-o", "--output_path", type=str, help="output path for predicting")
    parser.parse_args()

    args = parser.parse_args()
    return args

def predicting(args):
    valid_common_transform = get_valid_common_transform()
    valid_transform = get_valid_transform()
    valid_target_transform = get_valid_target_transform()
    valid_dataset = SegmentationDataset(
        args.input_path, 
        common_transform=valid_common_transform, 
        transform = valid_transform,
        target_transform=valid_target_transform,
        is_test=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    fcn16_1 = SegmentationWithFCN16(num_classes=NUM_CLASSES)
    fcn16_1.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg-fcn16_64.pkl')))
    fcn16_1.eval()
    fcn16_1 = fcn16_1.to(DEVICE)
    vgg16 = torchvision.models.vgg16()
    fcn32 = SegmentationWithFCN32(backbone=vgg16.features, num_classes=NUM_CLASSES)
    fcn32.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg-fcn32_65.pkl')))
    fcn32.eval()
    fcn32 = fcn32.to(DEVICE)

    vgg16_bn = torchvision.models.vgg16_bn()
    vgg16bn_fcn32 = SegmentationWithFCN32(backbone=vgg16_bn.features, num_classes=NUM_CLASSES)
    vgg16bn_fcn32.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg16bn-fcn32_65.pkl')))
    vgg16bn_fcn32.eval()
    vgg16bn_fcn32 = vgg16bn_fcn32.to(DEVICE)
    
    preds = None
    ori_height, ori_width = valid_dataset.ori_size
    with torch.no_grad():
        for idx, batch_data in enumerate(valid_loader):
            batch_imgs = batch_data
            
            batch_imgs = batch_imgs.to(DEVICE)
            outputs = []
            for model in [fcn16_1, fcn32, vgg16bn_fcn32]:
                output = model(batch_imgs)
                #output = torch.nn.functional.softmax(output, dim=1)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)
            output = outputs.mean(dim=0)
            _, pred = torch.max(output, 1)
            ### pred, class_masks are BATCH_SIZE,224,224
            ### According the mean_iou_evaluate.py, test_data should be 512,512
            pred = pred.unsqueeze(1)
            resized_pred = F.interpolate(
                pred.float(), size=(ori_height, ori_width)
            )
            
            preds = resized_pred.cpu().numpy() if preds is None else np.concatenate((preds, resized_pred.cpu().numpy()), axis=0)
        ### preds, labels are IMAGE_NUM,1,224,224
        preds = np.squeeze(preds)
        ret_onehot = maskclass2onehot(preds)
        ### ret_onehot are [IMAGE_NUM, 512, 512, 3]
        for i in range(ret_onehot.shape[0]):
            source_path = valid_dataset.datas[i]
            save_name = os.path.basename(source_path)
            save_path = os.path.join(args.output_path, save_name)
            print(save_path)
            img_tosave = Image.fromarray(np.uint8(255* ret_onehot[i]))
            img_tosave.save(os.path.join(args.output_path, save_name))
            
        """
        np.save('mock_preds.npy', preds)
        np.save('mock_labels.npy', labels)
        print(preds.shape, labels.shape)
        
        mean_iou = mean_iou_score(preds, labels)
        print('Mean_IOU: {:.4f}'.format(mean_iou))
        """

if __name__ == '__main__':
    args = parse_args()
    predicting(args)