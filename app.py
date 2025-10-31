import os
import sys
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path: sys.path.append(scripts_dir)

try:
    from inference import predict_image, load_model
    logger.info("✓ Inference functions imported.")
except (ImportError, AttributeError) as e:
    logger.critical(f"❌ CRITICAL ERROR: Could not import inference functions: {e}", exc_info=True)
    predict_image, load_model = None, None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_load_success = False
if load_model:
    logger.info("--- Attempting to load model on startup... ---")
    model_instance, _ = load_model()
    if model_instance is not None:
        model_load_success = True
        logger.info("✅ Model loaded successfully.")
    else:
        logger.critical("❌ CRITICAL: Model failed to load.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if not model_load_success:
        return jsonify({'error': 'Model is not loaded, cannot predict. Check server logs.'}), 503

    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files.get('file')
    if not file or file.filename == '': return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        predicted_class, confidence = predict_image(filepath)
        if "Error" in predicted_class:
            return jsonify({'error': 'Model prediction failed.'}), 500

        return jsonify({
            'filename': filename,
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%"
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Server error during processing.'}), 500
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_load_success})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
