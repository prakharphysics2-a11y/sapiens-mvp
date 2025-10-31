import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import json
import time
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

print("--- Loading Thermal PV Inference Module (Final Version) ---")

MODEL_DIR = "/var/data/model_weights"
MODEL_FILENAME = "resnet50_81_percent_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://www.dropbox.com/scl/fi/pmvdmnu3jjq379hh9b8xj/resnet50_81_percent_v1.pth?rlkey=3x197yyzs8m6t4vs19125gu&dl=1"

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = None
class_names = None
device = None

def download_model_if_needed(url, dest_path):
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_path):
        if not os.path.exists(dest_dir):
             logger.error(f"❌ ERROR: Destination directory {dest_dir} does not exist. Check Render disk mount path.")
             return False
        logger.info(f"Model not found. Downloading...")
        try:
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(
                desc=MODEL_FILENAME, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
            ) as bar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    bar.update(size)
            logger.info(f"✓ Model downloaded successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ ERROR: Download failed: {e}", exc_info=True)
            if os.path.exists(dest_path): os.remove(dest_path)
            return False
    else:
        logger.info(f"✓ Model already exists at {dest_path}.")
        return True

def load_model():
    global model, class_names, device
    if not download_model_if_needed(MODEL_URL, MODEL_PATH):
        return None, None

    device = torch.device("cpu") # Use CPU on Render free tier
    logger.info(f"--- Loading Model to {device} ---")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model_instance = models.resnet50(weights=None)
        num_ftrs = model_instance.fc.in_features
        loaded_class_names = checkpoint.get('class_names')
        if not loaded_class_names: raise ValueError("Class names not found in checkpoint!")

        num_classes = len(loaded_class_names)
        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.eval()
        model = model_instance.to(device)
        class_names = loaded_class_names
        logger.info("✓ Model ready for evaluation.")
        return model, class_names
    except Exception as e:
        logger.error(f"❌ ERROR: Failed during model loading!", exc_info=True)
        return None, None

def predict_image(image_path):
    if model is None: return "Inference Error - Model Not Loaded", None
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = img_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad(): outputs = model(batch_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        return class_names[top_catid.item()], top_prob.item()
    except Exception as e:
        logger.error(f"❌ ERROR during inference for {image_path}", exc_info=True)
        return "Inference Error", None
