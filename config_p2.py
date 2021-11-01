import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HW1_ROOT = os.path.join(BASE_DIR, 'hw1_data')
DATAROOT = os.path.join(HW1_ROOT, 'p2_data')
TRAIN_ROOT = os.path.join(DATAROOT, 'train')
VAL_ROOT = os.path.join(DATAROOT, 'validation')

IS_SAVE = False
SAVE_DIR = os.path.join(BASE_DIR, 'model_dict_p2_download')
#SAVE_DIR = os.path.join('/', os.path.join(os.path.join(os.path.join('tmp2', 'aislab'), 'hungjui'), 'model_dict_p2'))
#os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = 224
NUM_CLASSES = 7

EPOCH = 100
BATCH_SIZE = 32
NUM_WORKERS = 8
LR = 0.001
MOMENTUM = 0.9
#WEIGHT = torch.tensor([4.6046, 1., 7.3901, 5.0706, 19.9657, 7.3264, 9.8364])
WEIGHT = None

SCHEDULER_STEPSIZE = 7
SCHEDULER_GAMMA = 0.1
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MODEL_NUM = 4
if __name__ == '__main__':
    print(SAVE_DIR)
    """
    saved_models = os.listdir(SAVE_DIR)
    print(saved_models)
    if len(saved_models) > 3:
        min_acc = None
        model_to_removed = None
        for saved_model in saved_models:
            print(saved_model)
            acc = os.path.splitext(saved_model)[0].split('_')[1]
            acc = int(acc)
            if min_acc is None or acc < min_acc:
                min_acc = acc
                model_to_removed = saved_model
        print(model_to_removed)
        os.remove(os.path.join(SAVE_DIR, model_to_removed))
    """