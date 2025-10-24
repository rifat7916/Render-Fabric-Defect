import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template, url_for
from PIL import Image
import io
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL_PATH = 'fabric_defect_model.h5' # Assumes model is in the same directory
IMG_HEIGHT = 299
IMG_WIDTH = 299
# IMPORTANT: Ensure this order matches exactly how your model was trained!
CLASS_NAMES = ['hole', 'horizontal', 'verticle'] 

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load the trained Keras model ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        logging.info(f"Loading model from: {MODEL_PATH}")
        # Load model using tf.keras for better compatibility
        model = tf.keras.models.load_model(MODEL_PATH)
        # Optional: Warm up the model (makes the first prediction faster)
        dummy_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3))
        _ = model.predict(dummy_input)
        logging.info("✅ Model loaded and warmed up successfully.")
    else:
        logging.error(f"!!! Model file not found at {MODEL_PATH} !!!")
except Exception as e:
    logging.error(f"!!! Error loading model: {e} !!!", exc_info=True)
    # Keep model as None if loading fails

# --- Preprocessing Function ---
def preprocess_image(img_bytes):
    """Loads image from bytes, preprocesses it for the model."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Resize using LANCZOS for better quality if downscaling
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        # Rescale pixel values (must match training preprocessing)
        img_array = img_array / 255.0 
        logging.info("Image preprocessed successfully.")
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

# --- Define Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main HTML page."""
    logging.info("Serving index page.")
    # This will look for 'index.html' in the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, preprocessing, prediction, and returns result."""
    logging.info("Received prediction request.")
    if model is None:
        logging.error("Prediction attempt failed: Model not loaded.")
        return jsonify({'error': 'Model is not available or failed to load.'}), 500

    if 'file' not in request.files:
        logging.warning("Prediction failed: No file part in the request.")
        return jsonify({'error': 'No file part found in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        logging.warning("Prediction failed: No file selected.")
        return jsonify({'error': 'No file selected for upload.'}), 400

    if file:
        try:
            filename = file.filename # Get filename for logging
            logging.info(f"Processing uploaded file: {filename}")
            img_bytes = file.read()
            processed_img = preprocess_image(img_bytes)

            if processed_img is None:
                 logging.error(f"Image preprocessing failed for file: {filename}")
                 return jsonify({'error': 'Failed to process the uploaded image.'}), 400

            # --- Make prediction ---
            logging.info("Making prediction...")
            predictions = model.predict(processed_img)
            predicted_index = np.argmax(predictions[0])
            
            # Ensure index is within bounds
            if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
                 logging.error(f"Prediction index out of bounds: {predicted_index}")
                 return jsonify({'error': 'Model prediction resulted in an invalid class index.'}), 500
                 
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(np.max(predictions[0])) # Convert numpy float

            logging.info(f"Prediction successful: Class={predicted_class}, Confidence={confidence:.4f}")

            # Return prediction and confidence
            return jsonify({
                'prediction': predicted_class.capitalize(), # e.g., "Hole"
                'confidence': round(confidence * 100, 2) # e.g., 95.34
            })

        except Exception as e:
            logging.error(f"!!! Error during prediction process: {e} !!!", exc_info=True)
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

    logging.warning("Prediction failed: Unknown reason (file object likely invalid).")
    return jsonify({'error': 'An unknown error occurred during file processing.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Render uses the PORT environment variable; default to 8080 for local testing
    port = int(os.environ.get('PORT', 8080))
    # '0.0.0.0' makes it accessible within Render's network and locally
    logging.info(f"Starting Flask app on host 0.0.0.0, port {port}")
    # Set debug=False for production on Render
    app.run(host='0.0.0.0', port=port, debug=False) 
