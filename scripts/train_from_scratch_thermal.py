import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms, models
import sys, os, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model_setup import PVFaultDataset

print("=" * 70)
print("Training ResNet-50 FROM SCRATCH on Thermal Images")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use patches dataset (already classification format!)
DATA_DIR_TRAIN = 'data/patches/train'
DATA_DIR_VAL = 'data/patches/val'
MODEL_SAVE_PATH = 'models/thermal_from_scratch.pth'

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Aggressive augmentation for small dataset
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = PVFaultDataset(DATA_DIR_TRAIN, train_transform)
val_dataset = PVFaultDataset(DATA_DIR_VAL, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

num_classes = len(train_dataset.classes)
print(f"\nClasses: {train_dataset.classes}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

# Create ResNet-50 WITHOUT pretrained weights
model = models.resnet50(weights=None)

# Initialize weights properly for thermal images
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
print("=" * 70)
print("Starting Training")
print("=" * 70)

best_val_acc = 0.0
patience_counter = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Train
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    
    print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✓ Best: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 20:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

time_elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"Complete! Time: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"{'='*70}\n")

print(classification_report(val_labels, val_preds, target_names=train_dataset.classes, digits=4))
