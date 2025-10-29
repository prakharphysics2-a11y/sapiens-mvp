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

print("--- Thermal PV Inference Script (with Auto-Download) ---")

# --- Configuration ---
# Path on Render's persistent disk
MODEL_DIR = "/var/data/model_weights" # Standard persistent disk path on Render
MODEL_FILENAME = "resnet50_81_percent_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
# !!! REPLACE WITH YOUR ACTUAL DROPBOX LINK ENDING IN &dl=1 !!!
MODEL_URL = https://www.dropbox.com/scl/fi/pmvdmnu3jjq379hh9b8xj/resnet50_81_percent_v1.pth?rlkey=3xi97yyzs8w6t4vs19125gu47&st=nnvi8bjj&dl=1 

# Define the transformations
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("✓ Transforms defined.")

# --- Download Function ---
def download_model_if_needed(url, dest_path):
    """Downloads the model from URL to dest_path if it doesn't exist."""
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True) # Ensure directory exists

    if not os.path.exists(dest_path):
        print(f"Model not found at {dest_path}. Downloading from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=600) # Increased timeout
            response.raise_for_status() # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 # 1 Kibibyte

            # Use tqdm for progress bar
            with open(dest_path, 'wb') as f, tqdm(
                desc=MODEL_FILENAME,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)

            # Check if download was complete
            if total_size != 0 and os.path.getsize(dest_path) != total_size:
                 print("\n❌ ERROR: Download incomplete. File size mismatch.")
                 os.remove(dest_path) # Remove partial file
                 return False

            print(f"\n✓ Model downloaded successfully to {dest_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"\n❌ ERROR: Could not download model. Check URL and internet connection.")
            print(f"   Error: {e}")
            # Clean up potentially incomplete file
            if os.path.exists(dest_path):
                 try: os.remove(dest_path)
                 except OSError: pass
            return False
        except Exception as e:
            print(f"\n❌ ERROR: An unexpected error occurred during download: {e}")
            if os.path.exists(dest_path):
                 try: os.remove(dest_path)
                 except OSError: pass
            return False
    else:
        print(f"✓ Model already exists at {dest_path}. Skipping download.")
        return True # File already exists

# --- Load Model (Function called by Flask) ---
def load_model():
    """Downloads model if needed, then loads it."""
    global model, class_names, device # Make these global

    # --- Attempt to download the model first ---
    # Use MODEL_PATH defined globally
    if not download_model_if_needed(MODEL_URL, MODEL_PATH):
         print("❌ CRITICAL: Failed to obtain model file. Inference will fail.")
         return None, None # Indicate download/existence failure

    # --- Proceed with loading if download was successful or file existed ---
    print(f"\n--- Loading Model: {MODEL_PATH} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        print("✓ Checkpoint loaded.")

        model_instance = models.resnet50(weights=None)
        num_ftrs = model_instance.fc.in_features

        loaded_class_names = checkpoint.get('class_names')
        if not loaded_class_names:
            print("❌ ERROR: Class names not found in checkpoint!")
            loaded_class_names = ['crack', 'hotspot', 'no_anomaly', 'pid', 'shading', 'soiling'] # Fallback
            print(f"⚠️ Warning: Using manual class names: {loaded_class_names}")
        else:
            print(f"✓ Loaded class names: {loaded_class_names}")
        num_classes = len(loaded_class_names)

        model_instance.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
        print("✓ Architecture rebuilt.")

        model_instance.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Weights loaded.")

        model_instance.eval()
        model_instance = model_instance.to(device)
        print("✓ Model ready for evaluation.")

        model = model_instance
        class_names = loaded_class_names
        return model, class_names

    except Exception as e:
        print(f"❌ ERROR: Failed during model loading/building!")
        print(f"   Error: {e}")
        import traceback; traceback.print_exc()
        return None, None

# --- Global variables for loaded model ---
model = None
class_names = None
device = None

# --- Inference Function ---
# (Remains the same as before)
def predict_image(image_path):
    """Loads an image, preprocesses it, and returns the prediction."""
    global model, class_names, device # Use the globally loaded model

    if model is None or class_names is None or device is None:
         print("❌ ERROR: Model is not loaded. Cannot perform inference.")
         return "Inference Error - Model Not Loaded", None

    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}", None

    try:
        img = Image.open(image_path).convert('RGB')
        img_t = img_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        # Optional: Add timing here if desired
        # start_time = time.time() ... end_time = time.time() ...

        with torch.no_grad():
            outputs = model(batch_t)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        predicted_class = class_names[top_catid.item()]
        confidence = top_prob.item()

        print(f"✓ Inference complete for {os.path.basename(image_path)}")
        # print(f"⏱️ Inference Time: ... ms") # Add if timing included

        return predicted_class, confidence

    except Exception as e:
        print(f"❌ ERROR: Failed during inference for {image_path}")
        print(f"   Error details: {e}")
        return "Inference Error", None

# --- Main Execution (for local testing ONLY) ---
if __name__ == '__main__':
    print("\n--- Running Local Test Inference ---")

    # For local testing, save/load from current dir instead of /var/data
    MODEL_DIR = "." # Current directory
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

    model, class_names = load_model() # This will try to download if needed

    if model is None:
         print("Exiting due to model load failure.")
         sys.exit(1)

    if len(sys.argv) > 1: input_image_path = sys.argv[1]
    else: input_image_path = 'master_dataset/crack/crack_infrared_9.jpg' # Example

    if not os.path.exists(input_image_path):
         print(f"⚠️ Warning: Local test image '{input_image_path}' not found.")
         sys.exit(1)

    print(f"Input image path: {input_image_path}")
    predicted_class, confidence = predict_image(input_image_path)

    if predicted_class and confidence is not None:
        print("\n--- Prediction Result ---"); print(f"Image: {os.path.basename(input_image_path)}")
        print(f"Prediction: {predicted_class}"); print(f"Confidence: {confidence*100:.2f}%")
    else: print(f"\n--- Prediction Failed: {predicted_class} ---")
