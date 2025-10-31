import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms, models
import sys, os, time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model_setup import PVFaultDataset

print("=" * 80)
print("ULTIMATE TRAINING SCRIPT - Target: 90%+ Accuracy")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# === CRITICAL: VERIFY YOUR DATA FIRST ===
DATA_DIR_TRAIN = 'data/patches/train'
DATA_DIR_VAL = 'data/patches/val'

print("\n" + "=" * 80)
print("DATA VERIFICATION")
print("=" * 80)

# Check if data exists
if not os.path.exists(DATA_DIR_TRAIN):
    print(f"ERROR: {DATA_DIR_TRAIN} does not exist!")
    sys.exit(1)

# Count images per class
print("\nTraining Data Distribution:")
train_counts = {}
for class_name in os.listdir(DATA_DIR_TRAIN):
    class_path = os.path.join(DATA_DIR_TRAIN, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        train_counts[class_name] = count
        print(f"  {class_name:15s}: {count:4d} images")

total_train = sum(train_counts.values())
print(f"\n  TOTAL TRAIN: {total_train}")

print("\nValidation Data Distribution:")
val_counts = {}
for class_name in os.listdir(DATA_DIR_VAL):
    class_path = os.path.join(DATA_DIR_VAL, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        val_counts[class_name] = count
        print(f"  {class_name:15s}: {count:4d} images")

total_val = sum(val_counts.values())
print(f"\n  TOTAL VAL: {total_val}")

# === WARNING CHECKS ===
print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)

if total_train < 500:
    print(f"⚠️  WARNING: Only {total_train} training images. Recommended: 1000+")
    print("   → Dataset too small for 90%+ accuracy")
    
if total_val < 100:
    print(f"⚠️  WARNING: Only {total_val} validation images. Recommended: 200+")

min_class_size = min(train_counts.values())
max_class_size = max(train_counts.values())
if max_class_size > 5 * min_class_size:
    print(f"⚠️  WARNING: Severe class imbalance")
    print(f"   → Largest class: {max_class_size}, Smallest: {min_class_size}")
    print("   → Will use weighted sampling")

# User confirmation
print("\n" + "=" * 80)
response = input("Continue with training? (yes/no): ")
if response.lower() != 'yes':
    print("Training aborted.")
    sys.exit(0)

# === HYPERPARAMETERS ===
MODEL_SAVE_PATH = 'models/ultimate_model.pth'
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# === EXTREME DATA AUGMENTATION ===
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === LOAD DATA ===
train_dataset = PVFaultDataset(DATA_DIR_TRAIN, train_transform)
val_dataset = PVFaultDataset(DATA_DIR_VAL, val_transform)

num_classes = len(train_dataset.classes)
print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {train_dataset.classes}")

# === WEIGHTED SAMPLING FOR CLASS BALANCE ===
class_counts = Counter([label for _, label in train_dataset.samples])
class_weights = torch.FloatTensor([1.0 / class_counts[i] for i in range(num_classes)])
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

# === MODEL: PRETRAINED RESNET50 ===
print("\n" + "=" * 80)
print("MODEL SETUP")
print("=" * 80)

model = models.resnet50(weights='IMAGENET1K_V2')  # Use V2 (better weights)

# Freeze all layers except last 2 blocks
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (last block)
for param in model.layer4.parameters():
    param.requires_grad = True

# Custom classifier with dropout
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

model = model.to(device)

# Trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# === TRAINING SETUP ===
class_weights_tensor = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# === TRAINING LOOP ===
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80 + "\n")

best_val_acc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    running_loss = 0.0
    train_preds, train_labels = [], []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_loss = running_loss / len(train_dataset)
    train_acc = accuracy_score(train_labels, train_preds)
    
    # Validate
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print progress
    print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    scheduler.step()
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': train_dataset.classes,
            'history': history
        }, MODEL_SAVE_PATH)
        print(f"  ✓✓✓ NEW BEST: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) ✓✓✓")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= 30:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break
    
    # Unfreeze more layers after 50 epochs if accuracy < 70%
    if epoch == 50 and best_val_acc < 0.7:
        print("\n⚡ UNFREEZING LAYER3 FOR DEEPER FINE-TUNING ⚡\n")
        for param in model.layer3.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

time_elapsed = time.time() - start_time

# === FINAL RESULTS ===
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Total time: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print("=" * 80 + "\n")

# Classification report
print("Final Classification Report:")
print(classification_report(val_labels, val_preds, target_names=train_dataset.classes, digits=4))

# Confusion matrix
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_dataset.classes, 
            yticklabels=train_dataset.classes)
plt.title(f'Confusion Matrix - Val Acc: {best_val_acc:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved to: confusion_matrix.png")

# === DIAGNOSIS ===
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if best_val_acc >= 0.90:
    print("✅ SUCCESS! Achieved 90%+ accuracy")
elif best_val_acc >= 0.70:
    print("⚠️  MODERATE: 70-90% accuracy achieved")
    print("   → Need more data or better augmentation")
elif best_val_acc >= 0.50:
    print("❌ POOR: 50-70% accuracy")
    print("   → Dataset issues likely (mislabeling, insufficient data)")
else:
    print("❌ FAILURE: <50% accuracy")
    print("   → CRITICAL DATASET PROBLEM")
    print("   → Check:")
    print("      1. Are images corrupted?")
    print("      2. Are labels correct?")
    print("      3. Is dataset too small?")
    print("      4. Are images actually thermal/RGB as expected?")
