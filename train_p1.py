import os
import math
import torch
from config_p1 import EPOCH, TRAIN_ROOT, VAL_ROOT, BATCH_SIZE, NUM_WORKERS, DEVICE, SAVE_DIR
from dataset_p1 import ClassificationDataset, get_train_transform, get_valid_transform
from model_p1 import model, criterion, optimizer, lr_scheduler

def training(model):
    train_transform = get_train_transform()
    train_dataset = ClassificationDataset(TRAIN_ROOT, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_transform = get_valid_transform()
    valid_dataset = ClassificationDataset(VAL_ROOT, transform=valid_transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    model = model.to(DEVICE)
    best_acc = -1.0
    for epoch in range(EPOCH):
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
            running_corrects = 0
            for idx, batch_data in enumerate(loader):
                batch_imgs, batch_labels = batch_data
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(batch_imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, batch_labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item()* batch_imgs.shape[0]
                running_corrects += torch.sum(preds==batch_labels)
            if phase == 'train':
                lr_scheduler.step()
            ### print loss for train & validation
            if phase == 'train':
                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects.double() / len(train_dataset)
            else:
                epoch_loss = running_loss / len(valid_dataset)
                epoch_acc = running_corrects.double() / len(valid_dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                name_acc = str(math.floor(100*best_acc))
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vgg16_'+name_acc+'.pkl'))

if __name__ == '__main__':
    training(model)