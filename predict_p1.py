import os
import torch
import torchvision
import pandas as pd
import numpy as np
from config_p1 import EPOCH, BATCH_SIZE, NUM_WORKERS, DEVICE, SAVE_DIR, IS_PLOT
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
    valid_dataset = ClassificationDataset(args.input_path, transform=valid_transform, is_test=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    """
    vgg11_bn = torchvision.models.vgg11_bn()
    vgg11_bn.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg11_bn69.pkl')))
    vgg11_bn.eval()
    vgg11_bn = vgg11_bn.to(DEVICE)
    vgg13_bn = torchvision.models.vgg13_bn()
    vgg13_bn.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg13_bn73.pkl')))
    vgg13_bn.eval()
    vgg13_bn = vgg13_bn.to(DEVICE)
    """
    vgg16_bn = torchvision.models.vgg16_bn()
    vgg16_bn.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg16_bn78.pkl')))
    vgg16_bn.eval()
    vgg16_bn = vgg16_bn.to(DEVICE)
    vgg19_bn = torchvision.models.vgg19_bn()
    vgg19_bn.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'vgg19_bn77.pkl')))
    vgg19_bn.eval()
    vgg19_bn = vgg19_bn.to(DEVICE)
    """
    vgg16 = torchvision.models.vgg16()
    vgg16.load_state_dict(torch.load(os.path.join('model_vgg_p1', 'vgg16_71.pkl')))
    vgg16.eval()
    vgg16 = vgg16.to(DEVICE)
    resnet34 = torchvision.models.resnet34()
    resnet34.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet34_64.pkl')))
    resnet34.eval()
    resnet34 = resnet34.to(DEVICE)
    resnet50 = torchvision.models.resnet50()
    resnet50.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'resnet50_67.pkl')))
    resnet50.eval()
    resnet50 = resnet50.to(DEVICE)
    """

    preds = None
    feats_array = None
    #column_names = ["image_id", "label"]
    with torch.no_grad():
        for idx, batch_data in enumerate(valid_loader):
            batch_imgs = batch_data
            batch_imgs = batch_imgs.to(DEVICE)
            outputs = []
            features = []
            for model in [vgg16_bn, vgg19_bn]:
                output = model(batch_imgs)
                outputs.append(output)
                #output = torch.nn.functional.softmax(output, dim=1)
                ### get features of the second last layer
                if IS_PLOT:
                    feats = model.features(batch_imgs)
                    feats = model.avgpool(feats)
                    feats = torch.flatten(feats, 1)
                    feats = model.classifier[0](feats)
                    feats = model.classifier[1](feats)
                    ### feats is [32,4096]
                    features.append(feats)
            outputs = torch.stack(outputs, dim=0)
            outputs = outputs.mean(dim=0)
            _, ensemble_preds = torch.max(outputs, 1)
            preds = ensemble_preds if preds is None else torch.cat((preds, ensemble_preds), axis=0)
            if IS_PLOT:
                features = torch.stack(features, dim=0)
                features = features.mean(dim=0)
                feats_array = features.cpu().numpy() if feats_array is None else np.concatenate((feats_array, features.cpu().numpy()), axis=0)
        if IS_PLOT:
            print('feats_array', feats_array.shape)
            np.save('features.npy', feats_array)
        preds = preds.cpu().numpy().tolist()
        write_csv = {
            "image_id": [os.path.basename(valid_dataset.datas[i]) for i in range(len(preds))],
            "label": preds
        }
        write_csv = pd.DataFrame(write_csv)
        write_csv.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    args = parse_args()
    predicting(args)