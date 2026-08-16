import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import time


def get_vit_model(num_classes=10):
    # 1. Laden des ViT-B/16 mit ImageNet-Gewichten
    # 'B/16' bedeutet 'Base' Modell mit 16x16 Pixel Patches
    model = models.vit_b_16(weights=None)
    
    print(model)

    # 2. Den Classifier-Head anpassen
    # Bei ViT ist der finale Layer in 'model.heads.head'
    in_features = model.heads.head.in_features
    
    # Wir ersetzen den 1000-Klassen-Head durch einen für 10 Klassen (CIFAR-10)
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model.to(device)

def get_resnet18_model(num_classes=10):
    # Load ResNet50 with ImageNet weights
    model = models.resnet18(ResNet18_Weights.IMAGENET1K_V1)
    
    print(model)

    # adapt final layer for cifar classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model.to(device)

# Accuracy calculation function
def accuracy(output, target, topk=(1, 5)):
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Training loop function
def epoch_loop(model, db_loader, criterion, optimiser, epoch):
    is_train = optimiser is not None
    model.train() if is_train else model.eval()

    accuracies_top1 = []
    losses = []
    start_time = time.time()

    with torch.set_grad_enabled(is_train):
        for batch_ind, (img, target) in enumerate(db_loader):
            img, target = img.to(device), target.to(device)
            
            output = model(img)
            loss = criterion(output, target)

            # Accuracy calculation
            top1, _ = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            accuracies_top1.append(top1.item())

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            if is_train and (batch_ind + 1) % 100 == 0:
                print(f"  [Batch {batch_ind+1}/{len(db_loader)}] Loss: {np.mean(losses):.4f} | Top-1: {np.mean(accuracies_top1):.2f}%")

    return {'loss': np.mean(losses), 'top1': np.mean(accuracies_top1), 'time': time.time() - start_time}
    

# data transformation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)

transform_train = T.Compose([
    T.Resize((224, 224),  interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(),
    T.Normalize(mean, std)
])

transform_test = T.Compose([
    T.Resize((224, 224),  interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean, std)
])

trainloader = DataLoader(datasets.CIFAR10('data/', train=True, transform=transform_train, download=True), 
                         batch_size=32, shuffle=True, num_workers=4)
testloader = DataLoader(datasets.CIFAR10('data/', train=False, transform=transform_test, download=True), 
                        batch_size=32, shuffle=False, num_workers=4)

# start training
model_used = get_vit_model(num_classes=10)
epochs = 50
optimizer = optim.AdamW(model_used.parameters(), lr=5e-5, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
start_epoch = 0
# --- Training Configuration ---
last_checkpoint_path = "vit_last.pt"
best_model_path = "vit_untrained_supervised_run3.pt"

# 1. Load checkpoint if it exists to resume training
if os.path.exists(last_checkpoint_path):
    print(f"Loading checkpoint: {last_checkpoint_path}")
    checkpoint = torch.load(last_checkpoint_path)
    
    # Map the state to the current model, optimizer, and scheduler
    model_used.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Resuming from Epoch {start_epoch + 1} (Previous best accuracy: {best_acc:.2f}%)")
else:
    print("No checkpoint found. Starting training from scratch.")

# --- Training & Validation Loop ---
for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # Execute training and evaluation for the current epoch
    train_res = epoch_loop(model_used, trainloader, criterion, optimizer, epoch)
    val_res = epoch_loop(model_used, testloader, criterion, None, epoch)
    
    # Update learning rate based on the scheduler
    scheduler.step()
    
    # 2. Save the "Latest" state for potential resume
    # We save the full state including optimizer and scheduler for continuity
    checkpoint_state = {
        'epoch': epoch,
        'model_state_dict': model_used.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint_state, last_checkpoint_path)

    # 3. Track and save the "Best" model based on Validation Accuracy
    # If the current epoch outperforms previous ones, we save a dedicated 'best' file
    current_acc = val_res['top1']
    if current_acc > best_acc:
        best_acc = current_acc
        # We only save the state_dict here for easy inference later
        torch.save(model_used.state_dict(), best_model_path)
        print(f">>> New Best Performance! Acc: {best_acc:.2f}% | Model saved to {best_model_path}")

    print(f"Summary | Val Acc: {current_acc:.2f}% | Checkpoint updated.")

print(f"\nTraining completed. Highest accuracy reached: {best_acc}")