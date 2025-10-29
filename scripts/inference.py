import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import logging
import requests
from tqdm import tqdm
from pathlib import Path

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
MODEL_DIR = os.environ.get("MODEL_DIR", "/var/data/pv_hawk_models")
MODEL_FILENAME = "resnet50_81_percent_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://www.dropbox.com/scl/fi/pmvdmnu3jjq379hh9b8xj/resnet50_81_percent_v1.pth?rlkey=3x197yyzs8m6t4vs19125gu&dl=1"

# Image transforms
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

    # Check if file already exists
    if os.path.exists(dest_path):
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"✓ Model already exists: {dest_path} ({file_size_mb:.1f} MB)")
        return True

    logger.info(f"Model not found at {dest_path}. Downloading from Dropbox...")
    logger.info(f"URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Download size: {total_size / (1024*1024):.1f} MB")
        
        block_size = 1024 * 8
        
        # Download with progress bar
        with open(dest_path, 'wb') as f, tqdm(
            desc=MODEL_FILENAME, 
            total=total_size, 
            unit='iB',
            unit_scale=True, 
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                progress_bar.update(size)

        downloaded_size = os.path.getsize(dest_path)
        
        # Verify download
        if total_size != 0 and downloaded_size != total_size:
            logger.error(f"❌ Download incomplete. Expected {total_size}, got {downloaded_size}")
            os.remove(dest_path)
            return False

        logger.info(f"✓ Model downloaded successfully: {downloaded_size / (1024*1024):.1f} MB")
        return True
        
    except requests.exceptions.Timeout:
        logger.error("❌ Download timeout - took too long")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error("Check Dropbox URL is correct and ends with &dl=1")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during download: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

# --- Load Model Function ---
def load_model():
    """Load the fine-tuned ResNet50 model"""
    global model, class_names, device

    # Download model if needed
    if not download_model_if_needed(MODEL_URL, MODEL_PATH):
        logger.critical("❌ CRITICAL: Failed to obtain model file")
        return None, None

    logger.info(f"Loading model from: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"✓ Using device: {device}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        logger.info("✓ Checkpoint loaded")

        # Get class names
        loaded_class_names = checkpoint.get('class_names')
        if not loaded_class_names:
            loaded_class_names = ['Crack', 'Hotspot', 'Normal', 'PID', 'Shading', 'Soiling']
            logger.warning(f"Using fallback class names: {loaded_class_names}")
        else:
            logger.info(f"✓ Loaded class names: {loaded_class_names}")

        # Rebuild model architecture
        model_instance = models.resnet50(weights=None)
        num_ftrs = model_instance.fc.in_features
        num_classes = len(loaded_class_names)
        
        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        logger.info("✓ Architecture rebuilt")

        # Load weights
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Weights loaded")

        model_instance.eval()
        model_instance = model_instance.to(device)
        logger.info("✓ Model ready for inference")

        model = model_instance
        class_names = loaded_class_names
        return model, class_names

    except Exception as e:
        logger.error(f"❌ ERROR during model loading!", exc_info=True)
        return None, None

# --- Global variables ---
model = None
class_names = None
device = None

# Load model on module import
logger.info("Attempting to load the inference model on startup...")
model, class_names = load_model()

if model is None:
    logger.critical("❌ CRITICAL: load_model function returned None. Inference will fail.")

# --- Inference Function ---
def predict_image(image_path):
    """Run inference on a single image"""
    global model, class_names, device

    if model is None:
        logger.error("Model not loaded")
        return "Inference Error - Model Not Loaded", None

    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return "Error: Image file not found", None

    try:
        img = Image.open(image_path).convert('RGB')
        img_t = img_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        with torch.no_grad():
            outputs = model(batch_t)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)

        predicted_class = class_names[top_catid.item()]
        confidence = top_prob.item()

        logger.info(f"✓ Inference complete: {predicted_class} ({confidence*100:.2f}%)")
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"❌ ERROR during inference for {image_path}", exc_info=True)
        return "Inference Error", None

def get_model_info():
    """Return model metadata"""
    try:
        if model is None:
            return None
        
        total_params = sum(p.numel() for p in model.parameters())
        file_size = 0
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / 1e6  # MB

        return {
            'model_name': 'ResNet50 (Fine-tuned)',
            'device': str(device),
            'class_names': class_names,
            'num_classes': len(class_names) if class_names else 0,
            'input_size': (224, 224),
            'total_parameters': total_params,
            'model_file_size_mb': round(file_size, 2),
            'model_location': MODEL_PATH,
            'status': 'loaded'
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return None

# --- Local Testing ---
if __name__ == '__main__':
    print("-" * 60)
    print("PV Hawk Inference Module - Local Test")
    print("-" * 60)

    if model is None:
        print("❌ Model failed to load")
        sys.exit(1)

    print(f"✓ Model loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    print(f"Model path: {MODEL_PATH}")

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if os.path.exists(test_image):
            print(f"\nTesting with: {test_image}")
            pred, conf = predict_image(test_image)
            print(f"Prediction: {pred}")
            print(f"Confidence: {conf*100:.2f}%" if conf else "Failed")
        else:
            print(f"Image not found: {test_image}")
    else:
        print("\nUsage: python inference.py <image_path>")
