import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class PVFaultDataset(Dataset):
    """Custom dataset for PV fault images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Get all class directories
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        # Collect all image paths and labels
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load as grayscale then convert to 3-channel
        image = Image.open(img_path).convert('L')  # Grayscale
        image = image.convert('RGB')  # Convert to 3-channel for ResNet
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_custom_resnet(num_classes):
    """
    Create a ResNet-50 model adapted for thermal grayscale images.
    Uses standard 3-channel input (grayscale replicated across channels).
    """
    # Load pre-trained ResNet-50
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Freeze early layers, unfreeze layer3 and layer4 for fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Keep conv1 as 3-channel (grayscale will be replicated)
    # No need to modify conv1
    
    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# Default transform for thermal images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3-channel normalization
])
