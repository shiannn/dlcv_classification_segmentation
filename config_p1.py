import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW1_ROOT = os.path.join(BASE_DIR, 'hw1_data')
DATAROOT = os.path.join(HW1_ROOT, 'p1_data')
TRAIN_ROOT = os.path.join(DATAROOT, 'train_50')
VAL_ROOT = os.path.join(DATAROOT, 'val_50')

IS_PLOT = False
SAVE_DIR = os.path.join(BASE_DIR, 'model_dict_p1_download')
#os.makedirs(SAVE_DIR, exist_ok=True)

#print(TRAIN_ROOT, VAL_ROOT)
#print(os.listdir(VAL_ROOT))
EPOCH = 100
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 0.001
MOMENTUM = 0.9

SCHEDULER_STEPSIZE = 7
SCHEDULER_GAMMA = 0.1
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")