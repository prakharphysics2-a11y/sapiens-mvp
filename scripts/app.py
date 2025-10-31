import os
import sys
import time
import json
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the 'scripts' directory to the Python path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import inference function
predict_image = None
try:
    from inference import predict_image
    logger.info("✓ Inference function loaded successfully.")
except ImportError as e:
    logger.error(f"❌ ERROR: Could not import 'predict_image' from scripts/inference.py: {e}")

# Flask App Setup
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

# Create folders
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"✓ Folder ready: {folder}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(folder, max_age_hours=24):
    """Remove files older than max_age_hours to prevent disk bloat"""
    try:
        now = time.time()
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                if os.stat(filepath).st_mtime < now - (max_age_hours * 3600):
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

# Routes
@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handles image upload and fault prediction"""
    
    if predict_image is None:
        logger.error("Inference function not available")
        return jsonify({'error': 'Inference model not loaded. Check server logs.'}), 500

    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Save uploaded file
    original_filename = secure_filename(file.filename)
    filename_base, file_ext = os.path.splitext(original_filename)
    unique_filename = f"{filename_base}_{int(time.time() * 1000)}{file_ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)
        logger.info(f"✓ File saved: {filepath}")

        # Run inference
        logger.info(f"Running inference on: {filepath}")
        start_time = time.time()
        
        predicted_class, confidence = predict_image(filepath)
        
        inference_time = time.time() - start_time
        logger.info(f"✓ Inference complete in {inference_time:.2f}s: {predicted_class} ({confidence*100:.2f}%)")

        # Validate inference results
        if predicted_class is None or confidence is None or predicted_class == "Inference Error":
            logger.error("Inference returned invalid results")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Model prediction failed. Check server logs.'}), 500

        # Return results
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
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    finally:
        # Optional: Cleanup old files periodically
        if int(time.time()) % 100 == 0:  # Run cleanup occasionally
            cleanup_old_files(app.config['UPLOAD_FOLDER'])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predict_image is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch processing of multiple images"""
    
    if predict_image is None:
        return jsonify({'error': 'Inference model not loaded'}), 500

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    results = []
    
    for file in files[:20]:  # Limit to 20 files
        if not file or file.filename == '':
            continue

        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': 'File type not allowed'
            })
            continue

        original_filename = secure_filename(file.filename)
        filename_base, file_ext = os.path.splitext(original_filename)
        unique_filename = f"{filename_base}_{int(time.time() * 1000)}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)
            predicted_class, confidence = predict_image(filepath)

            if predicted_class and confidence is not None:
                results.append({
                    'filename': unique_filename,
                    'status': 'success',
                    'prediction': predicted_class,
                    'confidence': f"{confidence*100:.2f}%"
                })
            else:
                results.append({
                    'filename': unique_filename,
                    'status': 'error',
                    'error': 'Prediction failed'
                })
                
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': str(e)
            })
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({
        'total': len(results),
        'results': results
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size: 50MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("\n" + "="*50)
    logger.info("Starting Flask Server")
    logger.info("Access dashboard at: http://127.0.0.1:5000")
    logger.info("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
