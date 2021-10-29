import os
import math
import torch
from config_p1 import EPOCH, TRAIN_ROOT, VAL_ROOT, BATCH_SIZE, NUM_WORKERS, DEVICE, SAVE_DIR
from dataset_p1 import ClassificationDataset, get_train_transform, get_valid_transform
from model_p1 import models, criterion, optimizers, lr_schedulers, names
from timm.data.auto_augment import rand_augment_transform

def training():
    #train_transform = get_train_transform()
    tfm = rand_augment_transform(
        config_str='rand-m9-mstd0.5',
        hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
    )
    second_transform = get_train_transform()
    train_dataset = ClassificationDataset(TRAIN_ROOT, transform=tfm, second_transform=second_transform)
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
    for idx in range(len(models)):
        models[idx] = models[idx].to(DEVICE)
    best_accs = [0.72 for _ in models]
    for epoch in range(EPOCH):
        for phase in ['train', 'val']:
            if phase == 'train':
                print('[train]')
                for idx in range(len(models)):
                    models[idx].train()
                loader = train_loader
            else:
                print('[valid]')
                for idx in range(len(models)):
                    models[idx].eval()
                loader = valid_loader
            
            running_losses = [0.0 for _ in models]
            running_correctses = [0 for _ in models]
            outputs = [None for _ in models]
            ensemble_loss = 0.0
            ensemble_corrects = 0
            for idx, batch_data in enumerate(loader):
                batch_imgs, batch_labels = batch_data
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                for idx, (optimizer, model) in enumerate(zip(optimizers, models)):
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs[idx] = model(batch_imgs)
                        _, preds = torch.max(outputs[idx], 1)
                        loss = criterion(outputs[idx], batch_labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                
                    running_losses[idx] += loss.item()* batch_imgs.shape[0]
                    running_correctses[idx] += torch.sum(preds==batch_labels)
                ### calculate ensemble loss
                ensemble_outputs = torch.stack(outputs, dim=0)
                ensemble_outputs = ensemble_outputs.mean(dim=0)
                _, ensemble_preds = torch.max(ensemble_outputs, 1)
                temp_loss = criterion(ensemble_outputs, batch_labels)
                ensemble_loss += temp_loss.item()* batch_imgs.shape[0]
                ensemble_corrects += torch.sum(ensemble_preds==batch_labels)

            if phase == 'train':
                for lr_scheduler in lr_schedulers:
                    lr_scheduler.step()
            ### print loss for train & validation
            for idx, (name, running_loss, running_corrects) in enumerate(zip(
                names, running_losses, running_correctses)):
                if phase == 'train':
                    epoch_loss = running_loss / len(train_dataset)
                    epoch_acc = running_corrects.double() / len(train_dataset)
                else:
                    epoch_loss = running_loss / len(valid_dataset)
                    epoch_acc = running_corrects.double() / len(valid_dataset)
                print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(phase, name, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc > best_accs[idx]:
                    best_accs[idx] = epoch_acc
                    name_acc = str(math.floor(100*best_accs[idx]))
                    torch.save(models[idx].state_dict(), os.path.join(SAVE_DIR, name+name_acc+'.pkl'))
            if phase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, ensemble_loss/len(train_dataset), ensemble_corrects.double()/len(train_dataset))
                )
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, ensemble_loss/len(valid_dataset), ensemble_corrects.double()/len(valid_dataset))
                )

if __name__ == '__main__':
    training()