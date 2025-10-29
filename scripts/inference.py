import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import json
import time
import requests # For downloading the model
from tqdm import tqdm # For download progress bar
import logging # Use logging

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers: # Avoid adding multiple handlers if reloaded
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

print("--- Loading Thermal PV Inference Module ---")

# --- Configuration ---
MODEL_DIR = os.environ.get("MODEL_DIR", "/var/data/model_weights") # Standard persistent disk path on Render
MODEL_FILENAME = "resnet50_81_percent_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
# !!! ENSURE THIS IS YOUR CORRECT DROPBOX LINK ENDING IN &dl=1 !!!
MODEL_URL = "https://www.dropbox.com/scl/fi/pmvdmnu3jjq379hh9b8xj/resnet50_81_percent_v1.pth?rlkey=3x197yyzs8m6t4vs19125gu&dl=1" # YOUR LINK

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
logger.info("✓ Transforms defined.")

# --- Download Function ---
def download_model_if_needed(url, dest_path):
    """Downloads the model from URL to dest_path if it doesn't exist."""
    dest_dir = os.path.dirname(dest_path)
    # Create directory if it doesn't exist
    try:
        os.makedirs(dest_dir, exist_ok=True)
        logger.info(f"✓ Model directory ready: {dest_dir}")
    except PermissionError:
        logger.error(f"❌ Permission denied creating {dest_dir}")
        return False
    except Exception as e:
        logger.error(f"❌ Error creating directory: {e}")
        return False
    # --- THIS LINE IS COMMENTED OUT ---
    # os.makedirs(dest_dir, exist_ok=True) # Render provides the mount path, don't create parent
    # --- END OF CHANGE ---

    if not os.path.exists(dest_path):
        # Check if the directory itself exists first
        if not os.path.exists(dest_dir):
             logger.error(f"❌ ERROR: Destination directory {dest_dir} does not exist. Cannot download model.")
             return False

        logger.info(f"Model not found at {dest_path}. Downloading from URL...")
        try:
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 8

            with open(dest_path, 'wb') as f, tqdm(
                desc=MODEL_FILENAME, total=total_size, unit='iB',
                unit_scale=True, unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data); bar.update(size)

            downloaded_size = os.path.getsize(dest_path)
            if total_size != 0 and downloaded_size != total_size:
                 logger.error(f"❌ ERROR: Download incomplete. Expected {total_size}, got {downloaded_size}.")
                 os.remove(dest_path)
                 return False

            logger.info(f"✓ Model downloaded successfully to {dest_path}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ ERROR: Download failed. Check URL/network. Error: {e}")
            if os.path.exists(dest_path): try: os.remove(dest_path); except OSError: pass
            return False
        except Exception as e:
            logger.error(f"❌ ERROR: Unexpected error during download: {e}")
            if os.path.exists(dest_path): try: os.remove(dest_path); except OSError: pass
            return False
    else:
        logger.info(f"✓ Model already exists at {dest_path}.")
        return True

# --- Load Model Function ---
def load_model():
    global model, class_names, device

    if not download_model_if_needed(MODEL_URL, MODEL_PATH):
         logger.critical("❌ CRITICAL: Failed to obtain model file.")
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
            logger.error("❌ ERROR: Class names not in checkpoint!")
            loaded_class_names = ['crack', 'hotspot', 'no_anomaly', 'pid', 'shading', 'soiling'] # Fallback
            logger.warning(f"Using manual class names: {loaded_class_names}")
        else:
            logger.info(f"✓ Loaded class names: {loaded_class_names}")
        num_classes = len(loaded_class_names)

        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
        logger.info("✓ Architecture rebuilt.")
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Weights loaded.")
        model_instance.eval()
        model_instance = model_instance.to(device)
        logger.info("✓ Model ready for evaluation.")

        model = model_instance
        class_names = loaded_class_names
        return model, class_names
    except Exception as e:
        logger.error(f"❌ ERROR: Failed during model loading!", exc_info=True)
        return None, None

# --- Global variables ---
model = None
class_names = None
device = None

# --- Inference Function ---
def predict_image(image_path):
    global model, class_names, device
    if model is None: return "Inference Error - Model Not Loaded", None
    if not os.path.exists(image_path): return f"Error: Image file not found at {image_path}", None
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = img_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad(): outputs = model(batch_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        predicted_class = class_names[top_catid.item()]
        confidence = top_prob.item()
        logger.info(f"✓ Inference complete for {os.path.basename(image_path)}")
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"❌ ERROR during inference for {image_path}", exc_info=True)
        return "Inference Error", None

# --- Main Execution Block (for local testing ONLY) ---
if __name__ == '__main__':
    print("-" * 50); print("Running inference.py directly for local testing"); print("-" * 50)
    MODEL_DIR = "." # Current directory for local test
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
    model, class_names = load_model()
    if model is None: sys.exit(1)
    if len(sys.argv) > 1: input_image_path = sys.argv[1]
    else: input_image_path = 'master_dataset/crack/crack_infrared_9.jpg' # Needs valid example name
    if not os.path.exists(input_image_path):
        print(f"Test image not found: {input_image_path}")
        # Try finding *any* image in master_dataset for testing
        found_test_image = False
        for root, _, files in os.walk('master_dataset'):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    input_image_path = os.path.join(root, file)
                    print(f"Using fallback test image: {input_image_path}")
                    found_test_image = True
                    break
            if found_test_image: break
        if not found_test_image:
             print("Could not find any test images in master_dataset folder.")
             sys.exit(1)

    print(f"Input image: {input_image_path}")
    pred, conf = predict_image(input_image_path)
    if pred and conf is not None: print(f"Prediction: {pred}, Confidence: {conf*100:.2f}%")
    else: print(f"Prediction Failed: {pred}")
