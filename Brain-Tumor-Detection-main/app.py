import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
tumor_info = None
class_names = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, class_names
    try:
        model_path = 'model/brain_model.h5'
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return False
        
        model = tf.keras.models.load_model(model_path)
        
        # Define class names (should match training)
        class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def load_tumor_info():
    """Load tumor information from JSON file"""
    global tumor_info
    try:
        with open('tumor_info.json', 'r') as f:
            tumor_info = json.load(f)
        logger.info("✅ Tumor info loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading tumor info: {str(e)}")
        return False

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')  # Convert to RGB if needed
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_tumor(image_path):
    """Predict tumor type from image"""
    global model, class_names, tumor_info
    
    if model is None:
        return {"error": "Model not loaded"}
    
    # Preprocess image
    img_array = preprocess_image(image_path)
    if img_array is None:
        return {"error": "Failed to preprocess image"}
    
    try:
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get predicted class name
        predicted_class = class_names[predicted_class_idx]
        
        # Get tumor information
        tumor_data = tumor_info.get(predicted_class, {})
        
        # Format response
        result = {
            "tumor_type": tumor_data.get("name", predicted_class.replace("_", " ").title()),
            "confidence": f"{confidence * 100:.1f}%",
            "confidence_raw": confidence,
            "symptoms": tumor_data.get("symptoms", "Information not available"),
            "treatment": tumor_data.get("treatment", "Information not available"),
            "cure_rate": tumor_data.get("cure_rate", "Information not available"),
            "severity": tumor_data.get("severity", "Information not available"),
            "description": tumor_data.get("description", "Information not available"),
            "color": tumor_data.get("color", "#6c757d"),
            "class_name": predicted_class
        }
        
        logger.info(f"✅ Prediction: {result['tumor_type']} ({result['confidence']})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Please upload an image file."}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"📁 File uploaded: {filename}")
        
        # Make prediction
        result = predict_tumor(filepath)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Add file info to result
        result["filename"] = filename
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in predict endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    tumor_info_status = "loaded" if tumor_info is not None else "not loaded"
    
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "tumor_info": tumor_info_status,
        "class_names": class_names
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Load model and tumor info on startup
    print("🚀 Starting Brain Tumor Detection System...")
    
    if not load_model():
        print("❌ Failed to load model. Please ensure model/brain_model.h5 exists.")
        exit(1)
    
    if not load_tumor_info():
        print("❌ Failed to load tumor info. Please ensure tumor_info.json exists.")
        exit(1)
    
    print("✅ System ready!")
    print("🌐 Starting Flask server at http://localhost:5000")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 