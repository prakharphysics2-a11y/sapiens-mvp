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

# Add scripts directory to path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# --- Import inference functions ---
predict_image_func = None
load_model_func = None
get_model_info_func = None
try:
    import inference
    predict_image_func = inference.predict_image
    load_model_func = inference.load_model
    get_model_info_func = inference.get_model_info
    logger.info("✓ Inference functions imported successfully.")
except (ImportError, AttributeError) as e:
    logger.error(f"❌ ERROR: Could not import inference functions: {e}", exc_info=True)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
# Using relative paths for uploads is better for Render
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"✓ Upload folder ready: {UPLOAD_FOLDER}")

# --- CRITICAL FIX: Load the Model on Startup ---
model_load_success = False
if load_model_func:
    logger.info("--- Attempting to load the inference model on startup... ---")
    model_instance, class_names_loaded = load_model_func()
    if model_instance is not None and class_names_loaded is not None:
        model_load_success = True
        logger.info("✅ Model loaded successfully and is ready for inference.")
    else:
        logger.critical("❌ CRITICAL: load_model function failed. Inference will not work.")
else:
    logger.critical("❌ CRITICAL: load_model function not imported. Inference will not work.")
# --- END OF FIX ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    model_info = None
    if get_model_info_func:
        model_info = get_model_info_func()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_load_success, # This now reports the TRUE status
        'model_info': model_info,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if not model_load_success:
        return jsonify({'error': 'Inference model is not available. Check server startup logs.'}), 503

    if 'file' not in request.files: return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename): return jsonify({'error': 'File type not allowed'}), 400

    original_filename = secure_filename(file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    unique_filename = f"{filename_base}_{int(time.time() * 1000)}{file_ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)
        logger.info(f"✓ File saved: {filepath}")
        start_time = time.time()
        predicted_class, confidence = predict_image_func(filepath)
        inference_time = time.time() - start_time
        logger.info(f"✓ Inference result: {predicted_class} ({confidence*100:.2f}%)")

        if "Error" in predicted_class or confidence is None:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({'error': 'Model prediction failed'}), 500

        return jsonify({
            'filename': unique_filename,
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%",
            'confidence_value': round(confidence * 100, 2),
            'inference_time': f"{inference_time:.2f}s",
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"ERROR during upload/prediction: {e}", exc_info=True)
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try: return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError: return jsonify({'error': 'File not found'}), 404

# (Other routes and error handlers from your previous code are fine)

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("Starting Flask Server - PV Hawk MVP")
    logger.info("="*60 + "\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
