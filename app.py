import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the model
print("model loading")
model_path = 'best_deepfake_model.keras'
model = load_model(model_path)
print("model loaded")

# Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))  # Adjust size as per model input
    img = img.astype("float32") / 255.0
    img_array = img.reshape((1, 128, 128, 3))  # Adjust shape if needed
    return img_array

# Define a route for the homepage
@app.route('/')
def index():
    return render_template('frontend.html')

# Define a route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save file temporarily
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)
    
    # Preprocess the image and make a prediction
    img_array = preprocess_image(img_path)
    
    if img_array is None:
        return jsonify({'error': 'Invalid image'})
    
    prediction = model.predict(img_array)[0][0]
    result = 'Real' if prediction >= 0.5 else 'Fake'
    
    # Return the prediction, confidence, and image URL
    image_url = f"/uploads/{file.filename}"
    return jsonify({
        'prediction': result,
        'probability': round(float(prediction), 2),
        'image_url': image_url
    })

# Serve the uploaded image from the 'uploads' folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
