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

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

print("--- Loading Thermal PV Inference Module ---")

# --- Configuration ---
MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/model_weights") # For Render's persistent disk
MODEL_FILENAME = "resnet50_81_percent_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
# Your Dropbox direct download link
MODEL_URL = "https://www.dropbox.com/scl/fi/pmvdmnu3jjq379hh9b8xj/resnet50_81_percent_v1.pth?rlkey=3x197yyzs8m6t4vs19125gu&dl=1"

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
logger.info("✓ Transforms defined.")

# --- Global variables to hold the loaded model and info ---
model = None
class_names = None
device = None
model_info_dict = {"name": MODEL_FILENAME, "status": "Not Loaded"}

# --- Download Function ---
def download_model_if_needed(url, dest_path):
    dest_dir = os.path.dirname(dest_path)
    # On Render, the mount path itself exists, but we can't create its parent.
    # So we check if the mount path exists before proceeding.
    if not os.path.exists(dest_dir):
        logger.error(f"❌ ERROR: Destination directory {dest_dir} does not exist. Check Render disk mount path.")
        return False

    if not os.path.exists(dest_path):
        logger.info(f"Model not found at {dest_path}. Downloading from URL...")
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
            logger.error(f"❌ ERROR: Download failed. Error: {e}", exc_info=True)
            if os.path.exists(dest_path): os.remove(dest_path)
            return False
    else:
        logger.info(f"✓ Model already exists at {dest_path}.")
        return True

# --- Load Model Function ---
def load_model():
    global model, class_names, device, model_info_dict

    if not download_model_if_needed(MODEL_URL, MODEL_PATH):
        logger.critical("❌ CRITICAL: Failed to obtain model file.")
        model_info_dict['status'] = "Error - File Missing"
        return None, None

    logger.info(f"--- Loading Model: {MODEL_PATH} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"✓ Using device: {device}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        logger.info("✓ Checkpoint loaded.")

        model_instance = models.resnet50(weights=None)
        num_ftrs = model_instance.fc.in_features
        loaded_class_names = checkpoint.get('class_names')
        if not loaded_class_names:
            raise ValueError("Class names not found in checkpoint!")

        num_classes = len(loaded_class_names)
        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.eval()
        model = model_instance.to(device)
        class_names = loaded_class_names

        # Update model info for health check
        model_info_dict.update({
            "status": "Loaded",
            "device": str(device),
            "num_classes": num_classes,
            "classes": class_names,
            "accuracy": f"{checkpoint.get('val_acc', 0) * 100:.2f}%"
        })
        logger.info("✓ Model ready for evaluation.")
        return model, class_names
    except Exception as e:
        logger.error(f"❌ ERROR: Failed during model loading!", exc_info=True)
        model_info_dict['status'] = f"Error - {e}"
        return None, None

# --- Inference Function ---
def predict_image(image_path):
    global model
    if model is None: return "Inference Error - Model Not Loaded", None
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = img_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad(): outputs = model(batch_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        predicted_class = class_names[top_catid.item()]
        confidence = top_prob.item()
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"❌ ERROR during inference for {image_path}", exc_info=True)
        return "Inference Error", None

# --- New function for health check ---
def get_model_info():
    global model_info_dict
    return model_info_dict
