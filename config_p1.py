import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW1_ROOT = os.path.join(BASE_DIR, 'hw1_data')
DATAROOT = os.path.join(HW1_ROOT, 'p1_data')
TRAIN_ROOT = os.path.join(DATAROOT, 'train_50')
VAL_ROOT = os.path.join(DATAROOT, 'val_50')

#print(TRAIN_ROOT, VAL_ROOT)
#print(os.listdir(VAL_ROOT))