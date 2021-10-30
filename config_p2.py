import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW1_ROOT = os.path.join(BASE_DIR, 'hw1_data')
DATAROOT = os.path.join(HW1_ROOT, 'p2_data')
TRAIN_ROOT = os.path.join(DATAROOT, 'train')
VAL_ROOT = os.path.join(DATAROOT, 'validation')

IMAGE_SIZE = 224
NUM_CLASSES = 7

EPOCH = 100
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 0.001
MOMENTUM = 0.9

SCHEDULER_STEPSIZE = 7
SCHEDULER_GAMMA = 0.1
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")