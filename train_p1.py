import torch
from config_p1 import EPOCH, TRAIN_ROOT, BATCH_SIZE, NUM_WORKERS, DEVICE
from dataset_p1 import ClassificationDataset, get_train_transform
from model_p1 import model, criterion, optimizer, lr_scheduler

def training(model):
    train_transform = get_train_transform()
    classificationDataset = ClassificationDataset(TRAIN_ROOT, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        classificationDataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    model = model.to(DEVICE)
    for epoch in range(EPOCH):
        ### train
        model = model.train()
        for idx, batch_data in enumerate(train_loader):
            batch_imgs, batch_labels = batch_data
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            print('loss', loss)
        ### val
        exit(0)

if __name__ == '__main__':
    training(model)