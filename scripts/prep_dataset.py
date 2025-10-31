import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import pandas as pd  # For CSV if needed

# Config
input_dir = 'data/pvmd'
output_dir = 'data/patches'
classes = {0: 'PID', 1: 'soiling', 2: 'hotspot', 3: 'open_string', 4: 'normal'}
patch_size = 224
aug_factor = 15  # Boost for 15k+ patches

# Create dirs
for split in ['train', 'val', 'test']:
    for cls in classes.values():
        os.makedirs(f'{output_dir}/{split}/{cls}', exist_ok=True)

# Load PVMD (path, label)
all_data = []
for folder, label_map in [('Hotspots', 2), ('Cracks', 0), ('Shading', 3)]:
    folder_path = f'{input_dir}/{folder}'
    if not os.path.exists(folder_path):
        folder_path = f'{input_dir}/Shadings'  # Fallback
    for img_file in os.listdir(folder_path):
        if img_file.endswith(('.jpg', '.png')):
            all_data.append((f'{folder_path}/{img_file}', label_map))

print(f"Loaded {len(all_data)} base images from PVMD.")

# Synth PID/soiling (400 each for balance)
for i in range(400):  # PID synth: gradient degradation on cracks/hotspots
    base_path = np.random.choice([p for p, l in all_data if l in [0, 2]])
    base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
    gradient = np.linspace(0, 30, base_img.shape[1]).astype(np.uint8)
    pid_img = cv2.resize(base_img + gradient, (patch_size, patch_size))
    cv2.imwrite(f'{output_dir}/train/PID/synth_{i}.jpg', pid_img)
    all_data.append((f'{output_dir}/train/PID/synth_{i}.jpg', 0))
for i in range(400):  # Soiling synth: dust noise on normals/shadings
    base_path = np.random.choice([p for p, l in all_data if l in [4, 3]])
    base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
    noise = np.random.normal(0, 20, base_img.shape).astype(np.uint8)
    soiled_img = cv2.resize(cv2.add(base_img, noise), (patch_size, patch_size))
    cv2.imwrite(f'{output_dir}/train/soiling/synth_{i}.jpg', soiled_img)
    all_data.append((f'{output_dir}/train/soiling/synth_{i}.jpg', 1))

# Split (80/10/10, stratify)
train_data, temp_data = train_test_split(all_data, test_size=0.2, stratify=[l for _, l in all_data], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=[l for _, l in temp_data], random_state=42)

# Save base + aug train
for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    for path, label in data:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (patch_size, patch_size))
        cls_name = classes[label]
        base_name = os.path.basename(path)
        cv2.imwrite(f'{output_dir}/{split}/{cls_name}/{base_name}', img)
        if split == 'train' and aug_factor > 1:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.GaussianNoise(var_limit=(10, 50), p=0.5),  # Thermal noise
            ])
            for aug_i in range(1, aug_factor):
                aug = transform(image=img)
                aug_img = aug['image']
                cv2.imwrite(f'{output_dir}/{split}/{cls_name}/aug_{aug_i}_{base_name}', aug_img)

print(f"Prep done: Train ~{len(train_data)*aug_factor}, Val ~{len(val_data)}, Test ~{len(test_data)} patches")
for cls in classes.values():
    train_count = len([f for f in os.listdir(f'{output_dir}/train/{cls}') if f.endswith('.jpg')])
    print(f"{cls}: {train_count} train patches")
