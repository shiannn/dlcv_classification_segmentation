import os
from torch.utils.data import Dataset, DataLoader
from config_p1 import TRAIN_ROOT, VAL_ROOT
from PIL import Image
import torchvision.transforms as transforms

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        datas = []
        for img_name in os.listdir(self.root_dir):
            img_name_nosuffix = os.path.splitext(img_name)[0]
            class_label, image_id = img_name_nosuffix.split('_')
            class_label = int(class_label)
            img_name_abs = os.path.join(self.root_dir, img_name)
            datas.append((img_name_abs, class_label))
        self.datas = datas
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img_path, label = self.datas[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.datas)

def get_train_transform(jitter_param=0.4):
    transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
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

if __name__ == '__main__':
    train_transform = get_train_transform()
    classificationDataset = ClassificationDataset(TRAIN_ROOT, transform=train_transform)
    print(classificationDataset[25])