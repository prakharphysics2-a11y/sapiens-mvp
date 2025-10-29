import os
import sys
import time
import json
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging for clear server output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the 'scripts' directory to the Python path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# --- MODIFIED FOR DEPLOYMENT: Import BOTH functions ---
predict_image = None
load_model = None
try:
    from inference import predict_image, load_model
    logger.info("✓ Inference functions (predict_image, load_model) imported successfully.")
except ImportError as e:
    logger.error(f"❌ ERROR: Could not import functions from scripts/inference.py: {e}")

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"✓ Upload folder ready: {UPLOAD_FOLDER}")

# --- MODIFIED FOR DEPLOYMENT: Load Model ONCE on startup ---
logger.info("Attempting to load the inference model on startup...")
if load_model:
    model_loaded_successfully = load_model() # This will download the model if needed
else:
    model_loaded_successfully = False

if model_loaded_successfully:
    logger.info("✅✅✅ Model loaded and ready for inference.")
else:
    logger.critical("❌❌❌ CRITICAL: Model failed to load on startup. The /predict endpoints will fail.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handles single image upload and fault prediction"""
    if not model_loaded_successfully:
        logger.error("Inference function not available because model failed to load.")
        return jsonify({'error': 'Inference model not loaded. Check server logs.'}), 500

    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    original_filename = secure_filename(file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    unique_filename = f"{filename_base}_{int(time.time() * 1000)}{file_ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)
        logger.info(f"✓ File saved: {filepath}")

        logger.info(f"Running inference on: {filepath}")
        start_time = time.time()
        predicted_class, confidence = predict_image(filepath)
        inference_time = time.time() - start_time
        logger.info(f"✓ Inference complete in {inference_time:.2f}s: {predicted_class} ({confidence*100:.2f}%)")

        if confidence is None:
            logger.error("Inference returned invalid results (confidence is None)")
            return jsonify({'error': 'Model prediction failed. Check server logs.'}), 500

        return jsonify({
            'filename': unique_filename,
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%",
            'inference_time': f"{inference_time:.2f}s",
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"ERROR during upload/prediction: {e}", exc_info=True)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch processing of multiple images"""
    if not model_loaded_successfully:
        return jsonify({'error': 'Inference model not loaded'}), 500

    if 'files' not in request.files: return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    results = []
    for file in files[:20]:  # Limit to 20 files per batch
        if not file or not allowed_file(file.filename):
            if file and file.filename:
                results.append({'filename': file.filename, 'status': 'error', 'error': 'File type not allowed'})
            continue

        original_filename = secure_filename(file.filename)
        filename_base, file_ext = os.path.splitext(original_filename)
        unique_filename = f"{filename_base}_{int(time.time() * 1000)}_{len(results)}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)
            predicted_class, confidence = predict_image(filepath)
            if confidence is not None:
                results.append({
                    'filename': unique_filename, 'status': 'success',
                    'prediction': predicted_class, 'confidence': f"{confidence*100:.2f}%"
                })
            else:
                results.append({'filename': unique_filename, 'status': 'error', 'error': 'Prediction failed'})
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            results.append({'filename': file.filename, 'status': 'error', 'error': str(e)})
        finally:
            # Clean up individual files after processing
            if os.path.exists(filepath): os.remove(filepath)

    return jsonify({'total': len(results), 'results': results})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({ 'status': 'healthy', 'model_loaded': model_loaded_successfully })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size: 50MB'}), 413

# Run the App
if __name__ == '__main__':
    logger.info("\n" + "="*50)
    logger.info("Starting Flask Server for Sapiens Acceleration Tech MVP")
    logger.info("Access dashboard at: http://127.0.0.1:5000")
    logger.info("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
