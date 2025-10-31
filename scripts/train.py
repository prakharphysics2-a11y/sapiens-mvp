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
print("🚀 REVISITING EfficientNetV2-S (Lower LR) - Target: 90%+ Accuracy 🚀")
print("=" * 80)

# --- 1. CONFIGURATION ---
EPOCHS = 75              # EfficientNet usually trains faster
LEARNING_RATE = 0.0003   # <<< LOWERED LR for EfficientNetV2-S
BATCH_SIZE = 32          # Adjust based on VRAM if needed
DATA_DIR = 'master_dataset'
MODEL_SAVE_PATH = 'thermal_model_effnetv2s_lowLR_best.pth' # New name
PATIENCE = 15            # Standard patience
NUM_WORKERS = 4
SEED = 42

# --- 2. SETUP ---
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ Using device: {device}")
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, DATA_DIR)
print(f"✓ Data Path: {data_path}")
if not os.path.exists(data_path) or not os.listdir(data_path):
    print(f"\n❌ ERROR: Data directory '{data_path}' missing/empty!")
    exit(1)

# --- 3. DATA AUGMENTATION & TRANSFORMS ---
# (Using the same successful augmentations)
print("\n--- Applying Data Augmentation ---")
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(35),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("✓ Augmentations defined.")

# --- 4. LOAD DATASET & SPLIT ---
# (Same as before)
print("\n--- Loading and Splitting Dataset ---")
full_dataset = datasets.ImageFolder(data_path)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"✓ Found {len(full_dataset)} total images.")
print(f"✓ Classes ({num_classes}): {class_names}")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset_split, val_dataset_split = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(SEED))
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None): self.subset = subset; self.transform = transform
    def __getitem__(self, index): x, y = self.subset[index]; return self.transform(x) if self.transform else x, y
    def __len__(self): return len(self.subset)
train_dataset = TransformedDataset(train_dataset_split, transform=train_transforms)
val_dataset = TransformedDataset(val_dataset_split, transform=val_transforms)
print(f"✓ Split complete: Train={len(train_dataset)}, Validation={len(val_dataset)}")

# --- 5. WEIGHTED RANDOM SAMPLER ---
# (Same successful sampler setup)
print("\n--- Implementing Weighted Random Sampler ---")
train_indices = train_dataset_split.indices
train_targets = [full_dataset.targets[i] for i in train_indices]
class_sample_count = np.array([len(np.where(train_targets == t)[0]) for t in np.unique(train_targets)])
print("Class distribution (Training Set):")
for i, count in enumerate(class_sample_count): print(f"  {class_names[i]:15s}: {count:5d} images")
weight = 1. / np.maximum(class_sample_count, 1)
samples_weight = np.array([weight[t] for t in train_targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
print("✓ WeightedRandomSampler configured.")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print("✓ DataLoaders created.")

# --- 6. MODEL DEFINITION (EfficientNetV2-S Fine-tuning) --- ### <<-- USING EFFICIENTNET -->> ###
print("\n--- Building EfficientNetV2-S Model ---")
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
for param in model.parameters(): param.requires_grad = False
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True), # Use dropout rate appropriate for EfficientNet
    nn.Linear(num_ftrs, num_classes)
)
print(f"✓ Replaced final classifier layer for {num_classes} classes with Dropout.")
for param in model.classifier[1].parameters(): param.requires_grad = True # Only unfreeze the new layer
model = model.to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model ready. Trainable parameters: {trainable_params:,} / {total_params:,}")

# --- 7. LOSS, OPTIMIZER, SCHEDULER ---
print("\n--- Configuring Training Components ---")
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
print(f"✓ Loss: CrossEntropyLoss (Label Smoothing=0.1)")
# Use AdamW for just the trainable parameters (the final layer)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, # <<< Use the lowered LR
    weight_decay=0.01
)
print(f"✓ Optimizer: AdamW (LR={LEARNING_RATE}, Weight Decay=0.01)")
# Adjust T_max for the scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 100)
print(f"✓ Scheduler: CosineAnnealingLR (T_max={EPOCHS})")

# --- 8. TRAINING LOOP ---
# (Remains the same structure)
print("\n" + "=" * 80)
print(f"🚀 STARTING TRAINING FOR {EPOCHS} EPOCHS (PATIENCE={PATIENCE}) 🚀")
print("=" * 80 + "\n")
best_val_acc = 0.0
patience_counter = 0
start_time = time.time()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0; running_train_corrects = 0; batch_count = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs); loss = criterion(outputs, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _, preds = torch.max(outputs, 1)
        running_train_loss += loss.item() * inputs.size(0)
        running_train_corrects += torch.sum(preds == labels.data)
        batch_count += 1
    epoch_train_loss = running_train_loss / len(train_dataset) if len(train_dataset) > 0 else 0
    epoch_train_acc = running_train_corrects.double() / len(train_dataset) if len(train_dataset) > 0 else 0

    model.eval()
    running_val_loss = 0.0; running_val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs); loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_val_loss += loss.item() * inputs.size(0)
            running_val_corrects += torch.sum(preds == labels.data)
    epoch_val_loss = running_val_loss / len(val_dataset) if len(val_dataset) > 0 else 0
    epoch_val_acc = running_val_corrects.double() / len(val_dataset) if len(val_dataset) > 0 else 0

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {current_lr:.6f}")
    scheduler.step()

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc; save_path = os.path.join(base_dir, MODEL_SAVE_PATH)
        try:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                        'val_acc': best_val_acc.item(), 'class_names': class_names}, save_path)
            print(f"  ✨ NEW BEST! Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) - Model Saved ✨")
        except Exception as save_e: print(f"⚠️ Warning: Could not save model. Error: {save_e}")
        patience_counter = 0
    else:
        patience_counter += 1; print(f"  (Patience: {patience_counter}/{PATIENCE})")
    if patience_counter >= PATIENCE:
        print(f"\n⏳ Early stopping triggered at epoch {epoch+1}.")
        break

# --- 9. FINAL RESULTS ---
time_elapsed = time.time() - start_time
print("\n" + "=" * 80); print("🏁 TRAINING COMPLETE! 🏁"); print("=" * 80)
print(f"\n✓ Total Training Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"✓ Best model saved to: {os.path.join(base_dir, MODEL_SAVE_PATH)}")
if best_val_acc >= 0.90: print("\n🎉🎉🎉 EXCELLENT! Reached 90%+ target! Ready for Phase 3! 🎉🎉🎉")
elif best_val_acc >= 0.85: print("\n🚀 VERY GOOD! Achieved 85%+ accuracy! Strong candidate for Phase 3.")
elif best_val_acc >= 0.80: print("\n👍 SOLID RESULT! Achieved 80%+ accuracy.")
else: print(f"\n⚠️ BELOW TARGET ({best_val_acc*100:.1f}%). Model might require more tuning or ResNet-50 was better.")
print("\n" + "=" * 80)
