import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Load the model globally so it only loads once when the server starts
MODEL_PATH = "data/models/cotton_disease_model.keras"

try:
    print("--- [VISION] Loading MobileNetV2 Model... ---")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- [VISION] Model Loaded Successfully! ---")
except Exception as e:
    print(f"--- [VISION] Error loading model. Check path: {e} ---")
    model = None

# 2. These match the Kaggle Cotton Disease folder names exactly
CLASS_NAMES = [
    'diseased_cotton_leaf', 
    'diseased_cotton_plant', 
    'fresh_cotton_leaf', 
    'fresh_cotton_plant'
]

def analyze_crop_image(img_path: str) -> str:
    """
    Loads an image, formats it for MobileNetV2, and predicts the disease.
    """
    if model is None:
        return "Model not loaded"
        
    try:
        print(f"--- [VISION] Analyzing Image: {img_path} ---")
        
        # Load image and resize to what MobileNetV2 expects (224x224)
        img = image.load_img(img_path, target_size=(224, 224))
        
        # Convert to array and normalize pixels (0-1 range)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Run the prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Format the name beautifully (e.g., "Diseased Cotton Leaf")
        disease_name = CLASS_NAMES[predicted_index].replace("_", " ").title()
        
        print(f"--- [VISION] Result: {disease_name} ({confidence:.2f}%) ---")
        return disease_name
        
    except Exception as e:
        print(f"--- [VISION] Error analyzing image: {e} ---")
        return "Unknown"