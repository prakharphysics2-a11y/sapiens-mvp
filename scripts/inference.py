import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import json
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

print("--- Loading Thermal PV Inference Module (Git LFS Version) ---")

# --- Configuration ---
# The model is now a local file in our project
MODEL_DIR = "model_weights" 
MODEL_FILENAME = "resnet50_81_percent_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Standard transformations for the model
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Global variables to hold the loaded model ---
model = None
class_names = None
device = None
model_info_dict = {"name": MODEL_FILENAME, "status": "Not Loaded"}

# --- Load Model Function ---
def load_model():
    """Loads the model from the local LFS file."""
    global model, class_names, device, model_info_dict

    if not os.path.exists(MODEL_PATH):
        logger.critical(f"❌ CRITICAL: Model file not found at {MODEL_PATH}. Was it pushed with Git LFS?")
        model_info_dict['status'] = "Error - File Missing"
        return None, None

    # Use CPU for loading on free hosting tiers
    device = torch.device("cpu") 
    logger.info(f"--- Loading Model from {MODEL_PATH} to {device} ---")
    try:
        # Load the model with weights_only=False (for PyTorch 2.6+ fix)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False) 
        logger.info("✓ Checkpoint loaded.")

        model_instance = models.resnet50(weights=None)
        num_ftrs = model_instance.fc.in_features
        loaded_class_names = checkpoint.get('class_names')
        if not loaded_class_names: 
            raise ValueError("Class names not found in checkpoint!")

        num_classes = len(loaded_class_names)
        # Re-create the exact classifier head from training
        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.eval()
        model = model_instance.to(device)
        class_names = loaded_class_names

        # Update info for the /health check
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
    """Runs inference on a single image path."""
    if model is None: 
        return "Inference Error - Model Not Loaded", None
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = img_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad(): 
            outputs = model(batch_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        return class_names[top_catid.item()], top_prob.item()
    except Exception as e:
        logger.error(f"❌ ERROR during inference: {e}", exc_info=True)
        return "Inference Error", None

# --- Health Check Function ---
def get_model_info():
    """Returns the status of the loaded model."""
    return model_info_dict
