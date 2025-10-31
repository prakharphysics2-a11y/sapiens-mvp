import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter
import os
import time
import numpy as np
import random

print("=" * 80)
print("🔥 ADVANCED TRAINING - Target: 85%+ 🔥")
print("=" * 80)

# Configuration
EPOCHS = 100
WARMUP_EPOCHS = 10  # Warmup period
LEARNING_RATE = 0.0005
MIN_LR = 1e-6
BATCH_SIZE = 20  # Smaller for better gradient estimates
DATA_DIR = 'master_dataset'
MODEL_SAVE_PATH = 'thermal_model_advanced.pth'
PATIENCE = 25
SEED = 42

# Setup
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ Device: {device}")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, DATA_DIR)

if not os.path.exists(data_path):
    print(f"❌ ERROR: {data_path} not found!")
    exit(1)

# Augmentation - Even Stronger
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),  # More aggressive
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),  # Increased
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # Added
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(data_path)
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"✓ Total: {len(full_dataset)} images, {num_classes} classes")

# Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_split, val_split = random_split(full_dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(SEED))

# Apply transforms
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

train_dataset = TransformedDataset(train_split, train_transforms)
val_dataset = TransformedDataset(val_split, val_transforms)

# Weighted sampler
train_indices = train_split.indices
train_targets = [full_dataset.targets[i] for i in train_indices]
class_counts = np.array([len(np.where(np.array(train_targets) == t)[0]) for t in np.unique(train_targets)])

print("\nClass distribution (train):")
for i, count in enumerate(class_counts):
    print(f"  {class_names[i]:15s}: {count:5d}")

weight = 1. / class_counts
sample_weights = np.array([weight[t] for t in train_targets])
sample_weights = torch.from_numpy(sample_weights).double()

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)

# Model - Unfreeze MORE layers
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer2, layer3, layer4 (DEEPER fine-tuning)
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Enhanced classifier
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n✓ Trainable: {trainable:,} / {total:,} params")

# Loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Increased smoothing

# Optimizer with discriminative learning rates
optimizer = optim.AdamW([
    {'params': model.layer2.parameters(), 'lr': LEARNING_RATE / 30},
    {'params': model.layer3.parameters(), 'lr': LEARNING_RATE / 15},
    {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10},
    {'params': model.fc.parameters(), 'lr': LEARNING_RATE}
], weight_decay=0.02)

# Warmup + Cosine scheduler
def get_lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS  # Linear warmup
    else:
        # Cosine annealing after warmup
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)

print(f"\n✓ Loss: CrossEntropyLoss (smoothing=0.15)")
print(f"✓ Optimizer: AdamW (discriminative LR)")
print(f"✓ Scheduler: Warmup ({WARMUP_EPOCHS}) + CosineAnnealing")

# Training
print("\n" + "=" * 80)
print(f"🚀 TRAINING FOR {EPOCHS} EPOCHS 🚀")
print("=" * 80 + "\n")

best_val_acc = 0.0
patience_counter = 0
start_time = time.time()

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = running_loss / total_samples
    train_acc = correct / total_samples

    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    val_loss = val_loss / total_samples
    val_acc = correct / total_samples

    current_lr = optimizer.param_groups[-1]['lr']  # Get FC layer LR
    gap = train_acc - val_acc

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
          f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
          f"Gap: {gap:.4f} | LR: {current_lr:.6f}")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'class_names': class_names
        }, os.path.join(base_dir, MODEL_SAVE_PATH))
        print(f"  ✨ BEST: {val_acc:.4f} ({val_acc*100:.2f}%) ✨")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\n⏳ Early stop at epoch {epoch+1}")
        break

time_elapsed = time.time() - start_time

print("\n" + "=" * 80)
print("🏁 COMPLETE!")
print("=" * 80)
print(f"\n✓ Time: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
print(f"🏆 Best: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

if best_val_acc >= 0.85:
    print("\n🎉 SUCCESS! 85%+ achieved!")
elif best_val_acc >= 0.82:
    print("\n✅ VERY CLOSE! Consider Approach 3 (Ensemble)")
else:
    print("\n⚠️ Need more data or different architecture")
