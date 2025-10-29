import os
import sys
import time
import json
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging (remains the same)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add scripts directory (remains the same)
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path: sys.path.append(scripts_dir)

# --- Import inference functions ---
predict_image_func = None
load_model_func = None
try:
    # Import the module first
    import inference
    # Assign functions after successful import
    predict_image_func = inference.predict_image
    load_model_func = inference.load_model
    logger.info("✓ Inference functions imported successfully.")
except ImportError as e:
    logger.error(f"❌ ERROR: Could not import 'inference' module: {e}", exc_info=True)
except AttributeError as e:
     logger.error(f"❌ ERROR: Could not find functions in 'inference' module: {e}", exc_info=True)


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results' # Keep results folder if needed later
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]: os.makedirs(folder, exist_ok=True)
logger.info(f"✓ Folders ready: {UPLOAD_FOLDER}, {RESULTS_FOLDER}")

# --- Load Model during App Initialization ---
model_instance = None
class_names_loaded = None
model_load_success = False
if load_model_func:
    logger.info("Attempting to load the inference model on startup...")
    # Use a try-except block specifically for loading
    try:
        model_instance, class_names_loaded = load_model_func()
        if model_instance is not None and class_names_loaded is not None:
             model_load_success = True
             logger.info("✅ Model loaded successfully and ready for inference.")
        else:
             logger.critical("❌ CRITICAL: load_model function returned None. Inference endpoints will fail.")
    except Exception as load_e:
         logger.critical(f"❌ CRITICAL: Exception during model load: {load_e}", exc_info=True)
else:
    logger.critical("❌ CRITICAL: load_model function not imported. Inference endpoints will fail.")


# --- Utility Functions (remain the same) ---
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def cleanup_old_files(folder, max_age_hours=24):
    try:
        now = time.time()
        for filename in os.listdir(folder):
            fp = os.path.join(folder, filename);
            if os.path.isfile(fp) and os.stat(fp).st_mtime < now - (max_age_hours * 3600):
                os.remove(fp); logger.info(f"Cleaned up: {filename}")
    except Exception as e: logger.warning(f"Cleanup error: {e}")

# --- Routes ---
@app.route('/', methods=['GET'])
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    # Check if model is ready BEFORE processing
    if not model_load_success or predict_image_func is None:
        logger.error("Model not ready, cannot predict.")
        return jsonify({'error': 'Inference model is not available. Check server startup logs.'}), 503 # Service Unavailable

    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename): return jsonify({'error': 'File type not allowed'}), 400

    original_filename = secure_filename(file.filename)
    fb, fe = os.path.splitext(original_filename)
    unique_filename = f"{fb}_{int(time.time() * 1000)}{fe}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath); logger.info(f"✓ File saved: {filepath}")
        logger.info(f"Running inference on: {filepath}")
        start_time = time.time()
        # Use the imported function handle
        predicted_class, confidence = predict_image_func(filepath)
        inference_time = time.time() - start_time
        logger.info(f"✓ Inference result: {predicted_class} ({confidence*100:.2f}%)")

        if predicted_class is None or confidence is None or "Error" in predicted_class:
            logger.error(f"Inference failed for {unique_filename}. Result: {predicted_class}")
            if os.path.exists(filepath): os.remove(filepath) # Clean up file on failure
            return jsonify({'error': 'Model prediction failed.'}), 500

        return jsonify({'filename': unique_filename, 'prediction': predicted_class,
                        'confidence': f"{confidence*100:.2f}%", 'confidence_value': round(confidence * 100, 2),
                        'inference_time': f"{inference_time:.2f}s", 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"ERROR during upload/prediction for {unique_filename}: {e}", exc_info=True)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
         # Run cleanup less frequently
         if random.randint(1, 100) == 1: cleanup_old_files(app.config['UPLOAD_FOLDER'])

# (Other routes: /uploads/<filename>, /health, /batch-predict remain largely the same,
# just ensure they also check 'model_load_success' and use 'predict_image_func')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try: return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError: return jsonify({'error': 'File not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_load_success,
                    'timestamp': datetime.now().isoformat()})

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if not model_load_success or predict_image_func is None:
        return jsonify({'error': 'Inference model not loaded'}), 503
    # (Rest of batch logic is okay, ensure it uses predict_image_func)
    if 'files' not in request.files: return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    if not files or len(files) == 0: return jsonify({'error': 'No files selected'}), 400
    results = []
    for file in files[:20]:
        if not file or file.filename == '' or not allowed_file(file.filename):
             results.append({'filename': file.filename if file else 'N/A', 'status': 'error', 'error': 'Invalid file'})
             continue
        original_filename=secure_filename(file.filename); fb, fe=os.path.splitext(original_filename)
        unique_filename=f"{fb}_{int(time.time()*1000)}{fe}"; fp=os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(fp); pred_cls, conf = predict_image_func(fp) # Use func handle
            if pred_cls and conf is not None and "Error" not in pred_cls:
                results.append({'filename': unique_filename, 'status': 'success', 'prediction': pred_cls, 'confidence': f"{conf*100:.2f}%"})
            else: results.append({'filename': unique_filename, 'status': 'error', 'error': pred_cls or 'Prediction failed'})
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            results.append({'filename': file.filename, 'status': 'error', 'error': str(e)})
        finally: # Clean up batch files immediately
            if os.path.exists(fp): os.remove(fp)
    return jsonify({'total': len(results), 'results': results})

# Error Handlers (remain the same)
@app.errorhandler(413)
def too_large(e): return jsonify({'error': 'File too large. Max size: 50MB'}), 413
@app.errorhandler(500)
def internal_error(e): logger.error(f"Internal server error: {e}", exc_info=True); return jsonify({'error': 'Internal server error'}), 500

# Run App
if __name__ == '__main__':
    # Load model explicitly before running app
    if not model_load_success:
         logger.warning("Attempting model load again before starting server...")
         model_instance, class_names_loaded = load_model_func()
         if model_instance is not None:
              model_load_success = True
              logger.info("Model loaded successfully on second attempt.")
         else:
              logger.critical("Model failed to load on second attempt. Server starting without model.")

    logger.info("\n" + "="*50); logger.info("Starting Flask Server for Sapiens Acceleration Tech MVP")
    logger.info("Access dashboard at: http://127.0.0.1:5000"); logger.info("="*50 + "\n")
    # Turn off debug mode for slightly cleaner startup logs in Render, but keep reload
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=True)
