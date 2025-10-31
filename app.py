import os
import sys
import time
import random
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add scripts directory
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path: sys.path.append(scripts_dir)

# --- Import inference functions ---
predict_image_func = None
load_model_func = None
try:
    import inference
    predict_image_func = inference.predict_image
    load_model_func = inference.load_model
    logger.info("✓ Inference functions imported.")
except (ImportError, AttributeError) as e:
    logger.critical(f"❌ CRITICAL ERROR: Could not import inference functions: {e}", exc_info=True)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"✓ Upload folder ready: {UPLOAD_FOLDER}")

# --- Load Model on Startup ---
model_load_success = False
if load_model_func:
    logger.info("--- Attempting to load model on startup... ---")
    model_instance, _ = load_model_func()
    if model_instance is not None:
        model_load_success = True
        logger.info("✅ Model loaded successfully.")
    else:
        logger.critical("❌ CRITICAL: Model failed to load.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if not model_load_success:
        return jsonify({'error': 'Model is not loaded.'}), 503

    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files.get('file')
    if not file or file.filename == '': return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
         return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        predicted_class, confidence = predict_image_func(filepath)
        if "Error" in predicted_class:
            return jsonify({'error': 'Model prediction failed.'}), 500

        return jsonify({
            'filename': filename,
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%"
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Server error.'}), 500
    finally:
        if os.path.exists(filepath): os.remove(filepath)

# --- THIS IS THE MISSING ROUTE ---
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if not model_load_success:
        return jsonify({'error': 'Model is not loaded'}), 503
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    results = []
    for file in files[:20]: # Limit to 20 files per batch
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                predicted_class, confidence = predict_image_func(filepath)
                if predicted_class and "Error" not in predicted_class:
                    results.append({
                        'filename': filename,
                        'status': 'success',
                        'prediction': predicted_class,
                        'confidence': f"{confidence*100:.2f}%"
                    })
                else:
                    results.append({'filename': filename, 'status': 'error', 'error': 'Prediction failed'})
            except Exception as e:
                results.append({'filename': filename, 'status': 'error', 'error': 'Server processing error'})
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            results.append({'filename': file.filename, 'status': 'error', 'error': 'File type not allowed or invalid'})

    return jsonify({'total': len(results), 'results': results})
# --- END OF MISSING ROUTE ---

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_load_success})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
