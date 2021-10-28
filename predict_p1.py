import os
import torch
import torchvision
from config_p1 import EPOCH, BATCH_SIZE, NUM_WORKERS, DEVICE, SAVE_DIR
from dataset_p1 import ClassificationDataset, get_valid_transform
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path for predicting")
    parser.add_argument("-o", "--output_path", type=str, help="output path for predicting")
    parser.parse_args()

    args = parser.parse_args()
    return args

def predicting(args):
    valid_transform = get_valid_transform()
    valid_dataset = ClassificationDataset(args.input_path, transform=valid_transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    vgg16_bn = torchvision.models.vgg16_bn()
    vgg16_bn.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg16_bn74.pkl')))
    vgg16_bn.eval()
    vgg16_bn = vgg16_bn.to(DEVICE)
    vgg19_bn = torchvision.models.vgg19_bn()
    vgg19_bn.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg19_bn73.pkl')))
    vgg19_bn.eval()
    vgg19_bn = vgg19_bn.to(DEVICE)
    """
    vgg16 = torchvision.models.vgg16()
    vgg16.load_state_dict(torch.load(os.path.join('model_vgg_p1', 'vgg16_71.pkl')))
    vgg16.eval()
    vgg16 = vgg16.to(DEVICE)
    """
    resnet34 = torchvision.models.resnet34()
    resnet34.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet34_64.pkl')))
    resnet34.eval()
    resnet34 = resnet34.to(DEVICE)
    resnet50 = torchvision.models.resnet50()
    resnet50.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet50_67.pkl')))
    resnet50.eval()
    resnet50 = resnet50.to(DEVICE)

    ensemble_corrects = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(valid_loader):
            batch_imgs, batch_labels = batch_data
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            outputs = []
            for model in [vgg16_bn, vgg19_bn, resnet34, resnet50]:
                output = model(batch_imgs)
                #output = torch.nn.functional.softmax(output, dim=1)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)
            outputs = outputs.mean(dim=0)
            _, ensemble_preds = torch.max(outputs, 1)
            ensemble_corrects += torch.sum(ensemble_preds==batch_labels)
    print('Acc: {:.4f}'.format(ensemble_corrects.double()/len(valid_dataset)))

if __name__ == '__main__':
    args = parse_args()
    predicting(args)