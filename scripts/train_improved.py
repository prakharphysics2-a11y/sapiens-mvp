import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report
from tensorboardX import SummaryWriter
from torchvision import transforms
import sys
import os
import time
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model_setup import PVFaultDataset

print("=" * 70)
print("IMPROVED Thermal PV Fault Detection Training")
print("=" * 70)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR_TRAIN = 'data/thermal_final/train'
DATA_DIR_VAL = 'data/thermal_final/val'
MODEL_SAVE_PATH = 'models/thermal_improved.pth'
NUM_EPOCHS = 50
BATCH_SIZE = 16  # Smaller batch for better gradients
LEARNING_RATE = 3e-4  # Higher learning rate

print(f"Using device: {device}\n")

# Enhanced data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
train_dataset = PVFaultDataset(root_dir=DATA_DIR_TRAIN, transform=train_transform)
val_dataset = PVFaultDataset(root_dir=DATA_DIR_VAL, transform=val_transform)

num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

# Calculate class weights for imbalanced dataset
class_counts = Counter([label for _, label in train_dataset.samples])
class_weights = torch.FloatTensor([1.0 / class_counts[i] for i in range(num_classes)])
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(device)

print("Class distribution (train):")
for i, cls in enumerate(train_dataset.classes):
    print(f"  {cls:15s}: {class_counts[i]:4d} images (weight: {class_weights[i]:.4f})")

# Weighted sampler for balanced batches
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Create model - UNFREEZE MORE LAYERS
from torchvision import models

model = models.resnet50(weights='IMAGENET1K_V1')

# Unfreeze layer2, layer3, layer4 (more layers)
for param in model.parameters():
    param.requires_grad = False

for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Custom classifier
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

model = model.to(device)

# Weighted loss for class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                         steps_per_epoch=len(train_loader), 
                                         epochs=NUM_EPOCHS)

writer = SummaryWriter('runs/thermal_improved')
best_val_acc = 0.0
patience, patience_counter = 15, 0

print("\n" + "=" * 70)
print("Starting Training")
print("=" * 70 + "\n")

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    train_preds, train_labels = [], []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_loss = running_loss / len(train_dataset)
    train_acc = accuracy_score(train_labels, train_preds)
    
    # Validation
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'classes': train_dataset.classes
        }, MODEL_SAVE_PATH)
        print(f"  ✓ New best: {best_val_acc:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

writer.close()
time_elapsed = time.time() - start_time

print("\n" + "=" * 70)
print(f"Training Complete! Time: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
print(f"Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print("=" * 70 + "\n")

print("Per-Class Performance:")
print(classification_report(val_labels, val_preds, target_names=train_dataset.classes, digits=4))
