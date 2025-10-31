import os
import shutil
from pathlib import Path
from collections import Counter

print("=== Converting Thermal YOLO Dataset to Classification Format ===\n")

SOURCE_DIR = 'data/thermal_kaggle'
OUTPUT_DIR = 'data/thermal_final'

# Discover classes from labels
def discover_classes(split_name):
    labels_dir = os.path.join(SOURCE_DIR, split_name, 'labels')
    class_ids = []
    
    if not os.path.exists(labels_dir):
        return []
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.append(int(parts[0]))
    
    return class_ids

print("Discovering classes...")
train_classes = discover_classes('train')
valid_classes = discover_classes('valid')

all_classes = sorted(set(train_classes + valid_classes))
print(f"Found class IDs: {all_classes}\n")

class_counts = Counter(train_classes + valid_classes)
print("Class distribution in labels:")
for class_id in sorted(class_counts.keys()):
    print(f"  Class {class_id}: {class_counts[class_id]} detections")

# Class name mapping - generic for now
CLASS_NAMES = {
    0: 'Cell',
    1: 'Hot-Spot',
    2: 'Diode',
    3: 'Crack',
    4: 'Vegetation',
    5: 'Soiling',
    6: 'Shadowing',
    7: 'No-Anomaly'
}

print(f"\nUsing class names: {CLASS_NAMES}\n")

def parse_yolo_label(label_path):
    """Get primary class from YOLO label"""
    classes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
    except:
        return None
    
    if classes:
        return Counter(classes).most_common(1)[0][0]
    return None

def convert_split(split_name, output_split):
    images_dir = os.path.join(SOURCE_DIR, split_name, 'images')
    labels_dir = os.path.join(SOURCE_DIR, split_name, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"⚠ {images_dir} not found")
        return
    
    print(f"Converting {split_name} → {output_split}...")
    stats = Counter()
    processed = 0
    
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            class_id = parse_yolo_label(label_path)
            if class_id is not None:
                class_name = CLASS_NAMES.get(class_id, f'Class_{class_id}')
            else:
                class_name = 'No-Anomaly'
        else:
            class_name = 'No-Anomaly'
        
        # Create class directory
        class_dir = os.path.join(OUTPUT_DIR, output_split, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy image
        dst = os.path.join(class_dir, img_file)
        shutil.copy2(img_path, dst)
        stats[class_name] += 1
        processed += 1
    
    print(f"  ✓ Processed {processed} images")
    for class_name in sorted(stats.keys()):
        print(f"    {class_name}: {stats[class_name]}")
    print()

# Convert both splits
os.makedirs(OUTPUT_DIR, exist_ok=True)
convert_split('train', 'train')
convert_split('valid', 'val')

print("=" * 50)
print("✅ CONVERSION COMPLETE!")
print("=" * 50)
print(f"\nDataset location: {OUTPUT_DIR}/")
print("\nFinal distribution:")

for split in ['train', 'val']:
    split_path = os.path.join(OUTPUT_DIR, split)
    if os.path.exists(split_path):
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path)])
                print(f"  {class_name:15s}: {count:4d} images")
                total += count
        print(f"  {'TOTAL':15s}: {total:4d} images")

print("\n" + "=" * 50)
print("NEXT STEPS:")
print("=" * 50)
print("1. Update scripts/train.py:")
print("   DATA_DIR_TRAIN = 'data/thermal_final/train'")
print("   DATA_DIR_VAL = 'data/thermal_final/val'")
print("\n2. Run training:")
print("   python scripts/train.py")
